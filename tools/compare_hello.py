#!/usr/bin/env python3
import ast
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_F16 = ROOT / "llm-models" / "SmolLM2-135M-Instruct-f16.gguf"
DEFAULT_WEIGHTS_DIR = ROOT / "llm-models" / "weights_full"
DEFAULT_PROMPT_FILE = ROOT / "tb" / "prompt_ids.mem"
DEFAULT_PROMPT = "hello"

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


def ensure_f16(model_f16):
    if not model_f16.exists():
        raise RuntimeError(f"Missing model: {model_f16}")


def tokenize_prompt(model_f16, prompt, prompt_file):
    print(f"[compare] tokenize prompt: {prompt!r}")
    env = os.environ.copy()
    env["GGML_METAL_DISABLE"] = "1"
    cmd = [
        "llama-tokenize",
        "--log-disable",
        "--ids",
        "-m",
        str(model_f16),
        "-p",
        prompt,
    ]
    out = run(cmd, env=env)
    ids = ast.literal_eval(out.strip())
    print(f"[compare] prompt ids: {ids}")
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    with prompt_file.open("w", encoding="ascii") as f:
        for tid in ids:
            f.write(f"{tid:08x}\n")
    return ids


def eos_token_id(model_f16):
    if gguf is None:
        raise RuntimeError("Missing gguf package for eos_token_id()")
    reader = gguf.GGUFReader(str(model_f16))
    return int(reader.fields["tokenizer.ggml.eos_token_id"].contents())


def run_verilator(run_script, weights_dir, prompt_file, prompt_len, dump_topk=False):
    print(f"[compare] run verilator: {run_script}")
    cmd = [
        str(run_script),
        f"+weights_dir={weights_dir}",
        f"+prompt_ids={prompt_file}",
        f"+prompt_len={prompt_len}",
    ]
    if dump_topk:
        cmd.append("+dump_topk=1")
    out = run(cmd)
    topk = []
    scores = []
    for line in out.splitlines():
        if line.startswith("NEXT_TOKEN_ID="):
            next_id = int(line.split("=")[1])
        elif line.startswith("TOPK["):
            parts = line.split()
            tid = int(parts[1].split("=")[1])
            score = None
            for part in parts[2:]:
                if part.startswith("score_fp16="):
                    score = part.split("=")[1]
                    try:
                        score = int(score, 16)
                    except ValueError:
                        score = None
                    break
                if part.startswith("score_q=") or part.startswith("score="):
                    score = int(part.split("=")[1])
                    break
            topk.append((tid, score))
    if "next_id" not in locals():
        raise RuntimeError("Missing NEXT_TOKEN_ID from Verilator")
    topk_ids = [tid for tid, _ in topk] if topk else None
    return next_id, topk_ids


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


def _layer_norm(x, weight, bias, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    return norm * weight + bias


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def run_python_reference_gpt2(model_f16, prompt_ids, top_k=5):
    if gguf is None:
        return None
    print("[compare] python reference: GPT-2 path")
    reader = gguf.GGUFReader(str(model_f16))
    tensors = _tensor_map(reader)

    n_layer = _field_u32(reader, "gpt2.block_count")
    n_embd = _field_u32(reader, "gpt2.embedding_length")
    n_ff = _field_u32(reader, "gpt2.feed_forward_length")
    n_ctx = _field_u32(reader, "gpt2.context_length")
    n_head = _field_u32(reader, "gpt2.head_count") or _field_u32(reader, "gpt2.attention.head_count")
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
    logits = x @ out_weight
    top_ids = np.argsort(-logits)[:top_k].tolist()
    return int(top_ids[0]), top_ids


def _rms_norm(x, weight, eps):
    mean_sq = np.mean(x * x)
    return x / np.sqrt(mean_sq + eps) * weight


def _silu(x):
    return x / (1.0 + np.exp(-x))


def run_python_reference_llama(model_f16, prompt_ids, top_k=5):
    if gguf is None:
        return None
    print("[compare] python reference: LLaMA path")
    reader = gguf.GGUFReader(str(model_f16))
    tensors = _tensor_map(reader)

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
    kv_dim = n_kv_head * head_dim

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
            x_norm = _rms_norm(x, attn_norm_w[layer], eps)
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

            x_norm = _rms_norm(x, ffn_norm_w[layer], eps)
            gate = x_norm @ ffn_gate_w[layer]
            up = x_norm @ ffn_up_w[layer]
            act = _silu(gate) * up
            down = act @ ffn_dn_w[layer]
            x = x + down

    x = _rms_norm(x, out_norm_w, eps)
    logits = x @ token_embd
    top_ids = np.argsort(-logits)[:top_k].tolist()
    return int(top_ids[0]), top_ids


def _load_vocab(reader):
    tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    return [str(tok) for tok in tokens]


def _decode_ids(ids, vocab):
    return [vocab[tid] if 0 <= tid < len(vocab) else "<oob>" for tid in ids]


def run_llama_cli(model_f16, prompt):
    print("[compare] run llama.cpp (CPU)")
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
        str(model_f16),
        "-p",
        prompt,
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
        return eos_token_id(model_f16), "<eos>"
    tok_cmd = [
        "llama-tokenize",
        "--log-disable",
        "--ids",
        "--no-bos",
        "-m",
        str(model_f16),
        "-p",
        token_text,
    ]
    ids_out = run(tok_cmd, env=env)
    ids = ast.literal_eval(ids_out.strip())
    if len(ids) < 1:
        raise RuntimeError("Failed to tokenize llama-cli output")
    return ids[0], token_text


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare SV vs llama.cpp vs Python reference.")
    parser.add_argument("--model-f16", default=str(DEFAULT_MODEL_F16))
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--out", default=None, help="Write a text report to this path")
    parser.add_argument("--out-json", default=None, help="Write a JSON report to this path")
    parser.add_argument("--gpt2-fp16", action="store_true", help="Use FP16 GPT-2 SV model")
    args = parser.parse_args()

    model_f16 = Path(args.model_f16)
    weights_dir = Path(args.weights_dir)
    prompt_file = Path(args.prompt_file)
    prompt = args.prompt

    lines = []
    def emit(text):
        print(text)
        lines.append(text)

    emit("[compare] start")
    ensure_f16(model_f16)
    if not weights_dir.exists():
        raise RuntimeError(f"Missing weights dir: {weights_dir}")
    ids = tokenize_prompt(model_f16, prompt, prompt_file)
    vocab = None
    arch_name = None
    if gguf is not None:
        reader = gguf.GGUFReader(str(model_f16))
        vocab = _load_vocab(reader)
        arch = reader.fields.get("general.architecture")
        if arch is not None:
            arch_name = str(arch.contents())
    if arch_name == "llama":
        emit("[compare] architecture: llama")
        py_result = run_python_reference_llama(model_f16, ids)
        sv_token, sv_topk = run_verilator(
            ROOT / "tb" / "run_llama_infer.sh",
            weights_dir,
            prompt_file,
            len(ids),
            dump_topk=True,
        )
    else:
        emit("[compare] architecture: gpt2")
        py_result = run_python_reference_gpt2(model_f16, ids)
        run_script = ROOT / "tb" / ("run_gpt2_infer.sh" if args.gpt2_fp16 else "run_infer.sh")
        sv_token, sv_topk = run_verilator(
            run_script,
            weights_dir,
            prompt_file,
            len(ids),
            dump_topk=True,
        )
    llama_token, llama_text = run_llama_cli(model_f16, prompt)

    emit(f"prompt: {prompt}")
    emit(f"prompt_ids: {ids}")
    if py_result is not None:
        py_token, py_topk = py_result
        emit(f"python_next_token_id: {py_token}")
        emit(f"python_topk_ids: {py_topk}")
        if vocab is not None:
            emit(f"python_topk_tokens: {_decode_ids(py_topk, vocab)}")
    emit(f"sv_next_token_id: {sv_token}")
    if sv_topk is not None:
        emit(f"sv_topk_ids: {sv_topk}")
        if vocab is not None:
            emit(f"sv_topk_tokens: {_decode_ids(sv_topk, vocab)}")
    emit(f"llama_next_token_id: {llama_token}")
    emit(f"llama_next_token_text: {llama_text}")
    emit(f"sv_vs_llama: {'MATCH' if sv_token == llama_token else 'MISMATCH'}")
    if py_result is not None:
        emit(f"python_vs_llama: {'MATCH' if py_result[0] == llama_token else 'MISMATCH'}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    if args.out_json:
        import json

        report = {
            "model": str(model_f16),
            "weights_dir": str(weights_dir),
            "prompt": prompt,
            "prompt_ids": ids,
            "arch": arch_name,
            "python_next_token_id": py_result[0] if py_result is not None else None,
            "python_topk_ids": py_result[1] if py_result is not None else None,
            "sv_next_token_id": sv_token,
            "sv_topk_ids": sv_topk,
            "llama_next_token_id": llama_token,
            "llama_next_token_text": llama_text,
            "sv_vs_llama": sv_token == llama_token,
            "python_vs_llama": (py_result[0] == llama_token) if py_result is not None else None,
        }
        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
