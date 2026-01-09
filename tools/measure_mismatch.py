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


def run_verilator(weights_dir, ids):
    PROMPT_MEM.parent.mkdir(parents=True, exist_ok=True)
    with PROMPT_MEM.open("w", encoding="ascii") as f:
        for tid in ids:
            f.write(f"{tid:08x}\n")
    cmd = [
        str(ROOT / "tb" / "run_infer.sh"),
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


def _layer_norm(x, weight, bias, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    return norm * weight + bias


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def python_logits(model_path, prompt_ids):
    if gguf is None:
        raise RuntimeError("Missing gguf package")
    reader = gguf.GGUFReader(str(model_path))
    tensors = _tensor_map(reader)

    n_layer = _field_u32(reader, "gpt2.block_count")
    n_embd = _field_u32(reader, "gpt2.embedding_length")
    n_ff = _field_u32(reader, "gpt2.feed_forward_length")
    n_ctx = _field_u32(reader, "gpt2.context_length")
    n_head = _field_u32(reader, "gpt2.attention.head_count")
    if n_head is None:
        n_head = _field_u32(reader, "gpt2.head_count")
    if n_head is None:
        raise RuntimeError("Missing head count in GGUF metadata")
    eps = _field_f32(reader, "gpt2.attention.layer_norm_epsilon")
    if eps is None:
        eps = _field_f32(reader, "gpt2.layer_norm_epsilon", 1e-5)
    head_dim = n_embd // n_head

    token_embd = _tensor_data(tensors["token_embd.weight"])
    pos_embd = _tensor_data(tensors["position_embd.weight"])
    out_weight = _tensor_data(tensors["output.weight"])
    out_norm_w = _tensor_data(tensors["output_norm.weight"])
    out_norm_b = _tensor_data(tensors["output_norm.bias"])

    qkv_w = []
    qkv_b = []
    attn_norm_w = []
    attn_norm_b = []
    attn_out_w = []
    attn_out_b = []
    ffn_norm_w = []
    ffn_norm_b = []
    ffn_up_w = []
    ffn_up_b = []
    ffn_dn_w = []
    ffn_dn_b = []
    for layer in range(n_layer):
        prefix = f"blk.{layer}."
        attn_norm_w.append(_tensor_data(tensors[prefix + "attn_norm.weight"]))
        attn_norm_b.append(_tensor_data(tensors[prefix + "attn_norm.bias"]))
        qkv_w.append(_tensor_data(tensors[prefix + "attn_qkv.weight"]))
        qkv_b.append(_tensor_data(tensors[prefix + "attn_qkv.bias"]))
        attn_out_w.append(_tensor_data(tensors[prefix + "attn_output.weight"]))
        attn_out_b.append(_tensor_data(tensors[prefix + "attn_output.bias"]))
        ffn_norm_w.append(_tensor_data(tensors[prefix + "ffn_norm.weight"]))
        ffn_norm_b.append(_tensor_data(tensors[prefix + "ffn_norm.bias"]))
        ffn_up_w.append(_tensor_data(tensors[prefix + "ffn_up.weight"]))
        ffn_up_b.append(_tensor_data(tensors[prefix + "ffn_up.bias"]))
        ffn_dn_w.append(_tensor_data(tensors[prefix + "ffn_down.weight"]))
        ffn_dn_b.append(_tensor_data(tensors[prefix + "ffn_down.bias"]))

    k_cache = np.zeros((n_layer, n_ctx, n_head, head_dim), dtype=np.float32)
    v_cache = np.zeros((n_layer, n_ctx, n_head, head_dim), dtype=np.float32)

    for pos, tid in enumerate(prompt_ids):
        x = token_embd[:, tid] + pos_embd[:, pos]
        x = x.astype(np.float32)
        for layer in range(n_layer):
            ln1 = _layer_norm(x, attn_norm_w[layer], attn_norm_b[layer], eps)
            qkv = ln1 @ qkv_w[layer] + qkv_b[layer]
            q = qkv[:n_embd].reshape(n_head, head_dim)
            k = qkv[n_embd : 2 * n_embd].reshape(n_head, head_dim)
            v = qkv[2 * n_embd :].reshape(n_head, head_dim)

            k_cache[layer, pos] = k
            v_cache[layer, pos] = v

            attn_heads = np.zeros((n_head, head_dim), dtype=np.float32)
            scale = 1.0 / math.sqrt(head_dim)
            for h in range(n_head):
                k_hist = k_cache[layer, : pos + 1, h, :]
                v_hist = v_cache[layer, : pos + 1, h, :]
                scores = (k_hist @ q[h]) * scale
                scores = scores - scores.max()
                weights = np.exp(scores)
                weights = weights / weights.sum()
                attn_heads[h] = weights @ v_hist

            attn = attn_heads.reshape(n_embd)
            proj = attn @ attn_out_w[layer] + attn_out_b[layer]
            x = x + proj

            ln2 = _layer_norm(x, ffn_norm_w[layer], ffn_norm_b[layer], eps)
            up = ln2 @ ffn_up_w[layer] + ffn_up_b[layer]
            up = _gelu(up)
            down = up @ ffn_dn_w[layer] + ffn_dn_b[layer]
            x = x + down

    x = _layer_norm(x, out_norm_w, out_norm_b, eps)
    return x @ out_weight


def main():
    parser = argparse.ArgumentParser(description="Measure arithmetic mismatch between SV and float logits.")
    parser.add_argument("--model", required=True, help="Path to F16 GGUF model")
    parser.add_argument("--weights-dir", required=True, help="Path to exported Q8.8 weights")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--frac-w", type=int, default=8, help="Fractional bits for Q format")
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
    sv_next, topk = run_verilator(weights_dir, ids)
    logits = python_logits(model_path, ids)
    llama_next_id = None
    llama_next_text = None
    try:
        llama_next_id, llama_next_text = run_llama_next_token_id(model_path, args.prompt)
    except Exception as exc:
        print(f"llama.cpp error: {exc}", file=sys.stderr)

    frac = 1 << args.frac_w
    errors = []
    print(f"prompt: {args.prompt}")
    print(f"prompt_ids: {ids}")
    print(f"sv_next_token_id: {sv_next}")
    if llama_next_id is not None:
        print(f"llama_next_token_id: {llama_next_id}")
        print(f"llama_next_token_text: {llama_next_text}")
    print("id sv_q8_8 sv_float py_logit abs_err rel_err")
    for tid, score in topk:
        sv_float = score / frac
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
