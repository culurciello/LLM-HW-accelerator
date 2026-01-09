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
FRAC_W = 8

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


def parse_mem(path, fp16=False):
    data = []
    with path.open("r", encoding="ascii") as f:
        for line in f:
            val = int(line.strip(), 16)
            if fp16:
                data.append(val & 0xFFFF)
            else:
                if val & 0x8000:
                    val -= 0x10000
                data.append(val)
    arr = np.array(data, dtype=np.uint16 if fp16 else np.int16)
    if fp16:
        return arr.view(np.float16).astype(np.float32)
    return arr.astype(np.float32) / (1 << FRAC_W)


def write_ref(path, arr):
    with path.open("w", encoding="ascii") as f:
        for v in arr.flatten():
            f.write(f"{v:.6f}\n")


def _tensor_map(reader):
    return {getattr(t, "name", None): t for t in reader.tensors}


def _tensor_data(tensor):
    ttype = getattr(tensor, "tensor_type", None)
    if ttype == 8:
        return _dequant_q8_0(tensor)
    arr = np.array(tensor.data, dtype=np.float32)
    shape = list(getattr(tensor, "shape", []))
    if shape:
        arr = arr.reshape(shape)
    return arr


def _dequant_q8_0(tensor):
    data = np.asarray(tensor.data)
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    if data.ndim != 2:
        raise RuntimeError("Q8_0 tensor data expected 2D byte array")
    rows, row_bytes = data.shape
    block_bytes = 34
    if row_bytes % block_bytes != 0:
        raise RuntimeError(f"Q8_0 row bytes {row_bytes} not divisible by {block_bytes}")
    n_blocks = row_bytes // block_bytes
    out = np.empty((rows, n_blocks * 32), dtype=np.float32)
    for r in range(rows):
        row = data[r].reshape(n_blocks, block_bytes)
        scales = row[:, :2].copy().view("<f2").reshape(n_blocks).astype(np.float32)
        qs = row[:, 2:].view(np.int8).astype(np.float32)
        out[r] = (qs * scales[:, None]).reshape(-1)
    shape = list(getattr(tensor, "shape", []))
    if shape:
        if out.shape == tuple(shape):
            return out
        if out.T.shape == tuple(shape):
            return out.T
        if out.size == int(np.prod(shape)):
            return out.reshape(shape)
    return out


def quantize_q8_8(arr):
    scale = 1 << FRAC_W
    q = np.round(arr * scale).astype(np.int32)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q.astype(np.float32) / scale


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


def rms_norm(x, weight, eps):
    mean_sq = np.mean(x * x)
    return x / np.sqrt(mean_sq + eps) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def llama_layer_dump(model_path, prompt_ids, target_layer, dump_all, dump_pos):
    if gguf is None:
        raise RuntimeError("Missing gguf package")
    reader = gguf.GGUFReader(str(model_path))
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

    dumps = {}

    for pos, tid in enumerate(prompt_ids):
        x = token_embd[:, tid].astype(np.float32)
        for layer in range(n_layer):
            dump_this = (pos == dump_pos) and (dump_all or layer == target_layer)
            if dump_this:
                dumps["x_in"] = x.copy()

            x_norm = rms_norm(x, attn_norm_w[layer], eps)
            if dump_this:
                dumps["attn_norm"] = x_norm.copy()

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

            if dump_this:
                dumps["q"] = q.reshape(-1).copy()
                dumps["k"] = k.reshape(-1).copy()
                dumps["v"] = v.reshape(-1).copy()

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
            if dump_this:
                dumps["attn_out"] = attn.copy()

            proj = attn @ attn_out_w[layer]
            if dump_this:
                dumps["attn_proj"] = proj.copy()
            x = x + proj
            if dump_this:
                dumps["x_attn"] = x.copy()

            x_norm = rms_norm(x, ffn_norm_w[layer], eps)
            if dump_this:
                dumps["ffn_norm"] = x_norm.copy()

            gate = x_norm @ ffn_gate_w[layer]
            up = x_norm @ ffn_up_w[layer]
            if dump_this:
                dumps["ffn_gate"] = gate.copy()
                dumps["ffn_up"] = up.copy()

            act = silu(gate) * up
            if dump_this:
                dumps["ffn_act"] = act.copy()

            down = act @ ffn_dn_w[layer]
            if dump_this:
                dumps["ffn_out"] = down.copy()
            x = x + down
            if dump_this:
                dumps["x_out"] = x.copy()

    x = rms_norm(x, out_norm_w, eps)
    dumps["final_norm"] = x.copy()
    return dumps


def main():
    parser = argparse.ArgumentParser(description="Compare SV layer dumps to Python reference.")
    parser.add_argument("--model", required=True, help="Path to F16 GGUF model")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to dump")
    parser.add_argument("--dump-all", action="store_true", help="Dump all layers")
    parser.add_argument("--pos", type=int, default=-1, help="Token position (default: last)")
    parser.add_argument("--sv-dir", default=None, help="Path to SV dump dir (optional)")
    parser.add_argument("--ref-dir", default=str(ROOT / "tb" / "dump_ref"))
    parser.add_argument("--head", type=int, default=8, help="Print first N entries on mismatch")
    parser.add_argument("--sv-fp16", action="store_true", help="SV dumps are FP16 mems")
    parser.add_argument("--fp16", action="store_true", help="Quantize Python reference to FP16 before compare")
    parser.add_argument(
        "--quantize-ref",
        action="store_true",
        default=True,
        help="Quantize Python reference to Q8.8 before comparison (default: on)",
    )
    parser.add_argument(
        "--no-quantize-ref",
        dest="quantize_ref",
        action="store_false",
        help="Disable Q8.8 quantization of Python reference",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise RuntimeError(f"Missing model: {model_path}")

    ids = tokenize(model_path, args.prompt)
    if not ids:
        raise RuntimeError("Tokenization returned no IDs")
    dump_pos = args.pos if args.pos >= 0 else len(ids) - 1

    reader = gguf.GGUFReader(str(model_path))
    arch = reader.fields.get("general.architecture")
    arch_name = str(arch.contents()) if arch is not None else ""
    if arch_name != "llama":
        raise RuntimeError("compare_layer.py currently supports LLaMA/GQA models only")

    dumps = llama_layer_dump(model_path, ids, args.layer, args.dump_all, dump_pos)

    ref_dir = Path(args.ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    for tag, arr in dumps.items():
        write_ref(ref_dir / f"layer{args.layer}_{tag}.txt", arr)

    if args.sv_dir is None:
        print(f"Wrote reference dumps to {ref_dir}")
        return

    sv_dir = Path(args.sv_dir)
    if not sv_dir.exists():
        raise RuntimeError(f"Missing SV dump dir: {sv_dir}")

    if args.fp16:
        print("using FP16 reference for comparison")
    elif args.quantize_ref:
        print("using quantized Q8.8 reference for comparison")
    print("tag len max_abs mean_abs ref_min ref_max sv_min sv_max")
    for tag, ref in dumps.items():
        sv_path = sv_dir / f"layer{args.layer}_{tag}.mem"
        if not sv_path.exists():
            continue
        sv = parse_mem(sv_path, fp16=args.sv_fp16)
        if sv.shape[0] != ref.shape[0]:
            print(f"{tag} size_mismatch {sv.shape[0]} vs {ref.shape[0]}")
            continue
        if args.fp16:
            ref_cmp = ref.astype(np.float16).astype(np.float32)
        else:
            ref_cmp = quantize_q8_8(ref) if args.quantize_ref else ref
        diff = np.abs(ref_cmp - sv)
        print(
            f"{tag} {ref.shape[0]} {diff.max():.6f} {diff.mean():.6f} "
            f"{ref_cmp.min():.6f} {ref_cmp.max():.6f} {sv.min():.6f} {sv.max():.6f}"
        )
        if diff.max() > 1.0 and args.head > 0:
            head = args.head
            print(f"{tag} ref_head {ref_cmp[:head].tolist()}")
            print(f"{tag} sv_head  {sv[:head].tolist()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
