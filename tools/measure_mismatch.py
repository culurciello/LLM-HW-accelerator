#!/usr/bin/env python3
import argparse
import ast
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROMPT_MEM = ROOT / "tb" / "prompt_ids.mem"
DEFAULT_MODEL = ROOT / "llm-models" / "SmolLM2-135M-Instruct-f16.gguf"
DEFAULT_WEIGHTS = ROOT / "llm-models" / "weights_full"

try:
    import gguf
except ImportError:
    gguf = None


def run(cmd, cwd=ROOT, env=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return result.stdout


def tokenize(model_path, text):
    env = os.environ.copy()
    env["GGML_METAL_DISABLE"] = "1"
    cmd = [
        "llama-tokenize",
        "--log-disable",
        "--ids",
        "-m",
        str(model_path),
        "-p",
        text,
    ]
    out = run(cmd, env=env).strip()
    if not out:
        return []
    return ast.literal_eval(out)


def run_verilator(run_script, weights_dir, ids):
    PROMPT_MEM.parent.mkdir(parents=True, exist_ok=True)
    with PROMPT_MEM.open("w", encoding="ascii") as f:
        for tid in ids:
            f.write(f"{tid:08x}\n")
    cmd = [
        str(run_script),
        f"+weights_dir={weights_dir}",
        f"+prompt_ids={PROMPT_MEM}",
        f"+prompt_len={len(ids)}",
        "+dump_topk=1",
    ]
    out = run(cmd)
    topk = []
    next_id = None
    for line in out.splitlines():
        if line.startswith("NEXT_TOKEN_ID="):
            next_id = int(line.split("=")[1])
        elif line.startswith("TOPK["):
            parts = line.split()
            tid = int(parts[1].split("=")[1])
            score = int(parts[2].split("=")[1])
            topk.append((tid, score))
    if next_id is None:
        raise RuntimeError("Missing NEXT_TOKEN_ID from Verilator")
    return next_id, topk


def run_llama_next_token_id(model_path, prompt_text):
    env = os.environ.copy()
    env["GGML_METAL_DISABLE"] = "1"
    cmd = [
        "llama-cli",
        "--simple-io",
        "--no-display-prompt",
        "--single-turn",
        "--no-warmup",
        "--log-disable",
        "--device",
        "blas",
        "-m",
        str(model_path),
        "-p",
        prompt_text,
        "-n",
        "1",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--repeat-penalty",
        "1.0",
        "--presence-penalty",
        "0",
        "--frequency-penalty",
        "0",
        "--seed",
        "1",
        "-ngl",
        "0",
    ]
    text = run(cmd, env=env)
    lines = text.splitlines()
    prompt_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("> "):
            prompt_idx = idx
    if prompt_idx is None:
        token_text = text.strip("\n")
    else:
        gen_lines = []
        for line in lines[prompt_idx + 1 :]:
            if line.startswith("["):
                break
            gen_lines.append(line)
        token_text = "\n".join(gen_lines).strip("\n")
    if token_text == "":
        token_text = "<eos>"
    ids = tokenize(model_path, token_text)
    if not ids:
        raise RuntimeError("Failed to tokenize llama.cpp output")
    return ids[0], token_text


def _tensor_map(reader):
    return {getattr(t, "name", None): t for t in reader.tensors}


def _tensor_data(tensor):
    ttype = getattr(tensor, "tensor_type", None)
    if ttype not in (None, 0, 1):
        raise RuntimeError("Quantized GGUF tensors are not supported. Use the F16 model.")
    arr = np.array(tensor.data, dtype=np.float32)
    shape = list(getattr(tensor, "shape", []))
    if shape:
        arr = arr.reshape(shape)
    return arr


def _field_u32(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    return int(field.contents())


def _field_f32(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    return float(field.contents())


def python_logits(model_path, prompt_ids):
    if gguf is None:
        raise RuntimeError("Missing gguf package")
    reader = gguf.GGUFReader(str(model_path))
    tensors = _tensor_map(reader)

    arch = reader.fields.get("general.architecture")
    arch_name = str(arch.contents()) if arch is not None else ""
    if arch_name != "llama":
        raise RuntimeError("measure_mismatch.py supports LLaMA models only.")

    n_layer = _field_u32(reader, "llama.block_count")
    n_embd = _field_u32(reader, "llama.embedding_length")
    n_ff = _field_u32(reader, "llama.feed_forward_length")
    n_ctx = _field_u32(reader, "llama.context_length")
    n_head = _field_u32(reader, "llama.attention.head_count")
    n_kv_head = _field_u32(reader, "llama.attention.head_count_kv")
    rope_dim = _field_u32(reader, "llama.rope.dimension_count")
    rope_base = _field_f32(reader, "llama.rope.freq_base", 10000.0)
    eps = _field_f32(reader, "llama.attention.layer_norm_rms_epsilon", 1e-5)
    head_dim = n_embd // n_head

    token_embd = _tensor_data(tensors["token_embd.weight"])
    out_norm_w = _tensor_data(tensors["output_norm.weight"])

    attn_norm_w = []
    ffn_norm_w = []
    attn_q_w = []
    attn_k_w = []
    attn_v_w = []
    attn_out_w = []
    ffn_gate_w = []
    ffn_up_w = []
    ffn_dn_w = []
    for layer in range(n_layer):
        prefix = f"blk.{layer}."
        attn_norm_w.append(_tensor_data(tensors[prefix + "attn_norm.weight"]))
        ffn_norm_w.append(_tensor_data(tensors[prefix + "ffn_norm.weight"]))
        attn_q_w.append(_tensor_data(tensors[prefix + "attn_q.weight"]))
        attn_k_w.append(_tensor_data(tensors[prefix + "attn_k.weight"]))
        attn_v_w.append(_tensor_data(tensors[prefix + "attn_v.weight"]))
        attn_out_w.append(_tensor_data(tensors[prefix + "attn_output.weight"]))
        ffn_gate_w.append(_tensor_data(tensors[prefix + "ffn_gate.weight"]))
        ffn_up_w.append(_tensor_data(tensors[prefix + "ffn_up.weight"]))
        ffn_dn_w.append(_tensor_data(tensors[prefix + "ffn_down.weight"]))

    k_cache = np.zeros((n_layer, n_ctx, n_kv_head, head_dim), dtype=np.float32)
    v_cache = np.zeros((n_layer, n_ctx, n_kv_head, head_dim), dtype=np.float32)

    rope_pairs = rope_dim // 2
    inv_freq = 1.0 / (rope_base ** (np.arange(0, rope_pairs) * 2.0 / rope_dim))

    for pos, tid in enumerate(prompt_ids):
        x = token_embd[:, tid].astype(np.float32)
        for layer in range(n_layer):
            x_norm = x / np.sqrt(np.mean(x * x) + eps) * attn_norm_w[layer]
            q = x_norm @ attn_q_w[layer]
            k = x_norm @ attn_k_w[layer]
            v = x_norm @ attn_v_w[layer]

            q = q.reshape(n_head, head_dim)
            k = k.reshape(n_kv_head, head_dim)
            v = v.reshape(n_kv_head, head_dim)

            angles = pos * inv_freq
            cos = np.cos(angles)
            sin = np.sin(angles)
            for h in range(n_head):
                qh = q[h]
                q1 = qh[:rope_dim:2]
                q2 = qh[1:rope_dim:2]
                qh[:rope_dim:2] = q1 * cos - q2 * sin
                qh[1:rope_dim:2] = q1 * sin + q2 * cos
            for h in range(n_kv_head):
                kh = k[h]
                k1 = kh[:rope_dim:2]
                k2 = kh[1:rope_dim:2]
                kh[:rope_dim:2] = k1 * cos - k2 * sin
                kh[1:rope_dim:2] = k1 * sin + k2 * cos

            k_cache[layer, pos] = k
            v_cache[layer, pos] = v

            attn_heads = np.zeros((n_head, head_dim), dtype=np.float32)
            scale = 1.0 / math.sqrt(head_dim)
            group = n_head // n_kv_head
            for h in range(n_head):
                kv_h = h // group
                k_hist = k_cache[layer, : pos + 1, kv_h, :]
                v_hist = v_cache[layer, : pos + 1, kv_h, :]
                scores = (k_hist @ q[h]) * scale
                scores = scores - scores.max()
                weights = np.exp(scores)
                weights = weights / weights.sum()
                attn_heads[h] = weights @ v_hist

            attn = attn_heads.reshape(n_embd)
            proj = attn @ attn_out_w[layer]
            x = x + proj

            x_norm = x / np.sqrt(np.mean(x * x) + eps) * ffn_norm_w[layer]
            gate = x_norm @ ffn_gate_w[layer]
            up = x_norm @ ffn_up_w[layer]
            act = (gate / (1.0 + np.exp(-gate))) * up
            down = act @ ffn_dn_w[layer]
            x = x + down

    x = x / np.sqrt(np.mean(x * x) + eps) * out_norm_w
    return token_embd.T @ x


def fp16_bits_to_float(score):
    score16 = score & 0xFFFF
    return np.array([score16], dtype=np.uint16).view(np.float16).astype(np.float32)[0]


def main():
    parser = argparse.ArgumentParser(description="Measure arithmetic mismatch between SV and float logits.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to F16 GGUF model")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS), help="Path to exported FP16 weights")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    args = parser.parse_args()

    model_path = Path(args.model)
    weights_dir = Path(args.weights_dir)
    if not model_path.exists():
        raise RuntimeError(f"Missing model: {model_path}")
    if not weights_dir.exists():
        raise RuntimeError(f"Missing weights dir: {weights_dir}")

    ids = tokenize(model_path, args.prompt)
    if not ids:
        raise RuntimeError("Tokenization returned no IDs")

    run_script = ROOT / "tb" / "run_llama_infer.sh"
    sv_next, topk = run_verilator(run_script, weights_dir, ids)
    logits = python_logits(model_path, ids)
    llama_next_id = None
    llama_next_text = None
    try:
        llama_next_id, llama_next_text = run_llama_next_token_id(model_path, args.prompt)
    except Exception as exc:
        print(f"llama.cpp error: {exc}", file=sys.stderr)

    errors = []
    print(f"prompt: {args.prompt}")
    print(f"prompt_ids: {ids}")
    print(f"sv_next_token_id: {sv_next}")
    if llama_next_id is not None:
        print(f"llama_next_token_id: {llama_next_id}")
        print(f"llama_next_token_text: {llama_next_text}")
    print("id sv_fp16 sv_float py_logit abs_err rel_err")
    for tid, score in topk:
        sv_float = fp16_bits_to_float(score)
        py_logit = float(logits[tid])
        abs_err = abs(py_logit - sv_float)
        rel_err = abs_err / (abs(py_logit) + 1e-9)
        errors.append(abs_err)
        print(
            f"{tid} {score} {sv_float:.6f} {py_logit:.6f} {abs_err:.6f} {rel_err:.6f}"
        )
    if errors:
        mean_err = sum(errors) / len(errors)
        max_err = max(errors)
        print(f"mean_abs_err: {mean_err:.6f}")
        print(f"max_abs_err: {max_err:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
