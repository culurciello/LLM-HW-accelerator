#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError as exc:
    gguf = None
    _gguf_import_error = exc


def _require_gguf():
    if gguf is None:
        msg = (
            "Missing Python package 'gguf'. Install it with:\n"
            "  pip install gguf\n"
            "or\n"
            "  pip install llama-cpp-python\n"
        )
        raise RuntimeError(msg) from _gguf_import_error


def _get_tensor(reader, name):
    for t in reader.tensors:
        if getattr(t, "name", None) == name:
            return t
    raise KeyError(name)


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


def to_q8_8(arr, frac_w):
    scale = 1 << frac_w
    scaled = np.round(arr * scale).astype(np.int32)
    clipped = np.clip(scaled, -32768, 32767).astype(np.int16)
    return clipped


def write_mem(path, arr):
    with path.open("w", encoding="ascii") as f:
        for val in arr.flatten(order="C"):
            f.write(f"{int(val) & 0xFFFF:04x}\n")


def write_mem_fp16(path, arr):
    fp16 = np.asarray(arr, dtype=np.float16)
    with path.open("w", encoding="ascii") as f:
        for val in fp16.flatten(order="C").view(np.uint16):
            f.write(f"{int(val) & 0xFFFF:04x}\n")


def field_u32(reader, key):
    field = reader.fields[key]
    return int(field.contents())


def field_list(reader, key):
    field = reader.fields[key]
    return field.contents()


def export_gpt2(reader, out_dir, frac_w, fp16):
    n_layer = field_u32(reader, "gpt2.block_count")
    n_embd = field_u32(reader, "gpt2.embedding_length")
    n_ff = field_u32(reader, "gpt2.feed_forward_length")
    n_ctx = field_u32(reader, "gpt2.context_length")
    vocab = len(field_list(reader, "tokenizer.ggml.tokens"))

    token_embd = _tensor_data(_get_tensor(reader, "token_embd.weight"))
    pos_embd = _tensor_data(_get_tensor(reader, "position_embd.weight"))
    out_weight = _tensor_data(_get_tensor(reader, "output.weight"))
    out_norm_w = _tensor_data(_get_tensor(reader, "output_norm.weight"))
    out_norm_b = _tensor_data(_get_tensor(reader, "output_norm.bias"))

    if token_embd.shape != (n_embd, vocab):
        raise RuntimeError(f"token_embd shape {token_embd.shape}")
    if pos_embd.shape != (n_embd, n_ctx):
        raise RuntimeError(f"pos_embd shape {pos_embd.shape}")
    if out_weight.shape != (n_embd, vocab):
        raise RuntimeError(f"output.weight shape {out_weight.shape}")

    if fp16:
        write_mem_fp16(out_dir / "token_embd.mem", token_embd.T)
        write_mem_fp16(out_dir / "pos_embd.mem", pos_embd.T)
        write_mem_fp16(out_dir / "output_weight.mem", out_weight.T)
        write_mem_fp16(out_dir / "output_norm_weight.mem", out_norm_w)
        write_mem_fp16(out_dir / "output_norm_bias.mem", out_norm_b)
    else:
        write_mem(out_dir / "token_embd.mem", to_q8_8(token_embd.T, frac_w))
        write_mem(out_dir / "pos_embd.mem", to_q8_8(pos_embd.T, frac_w))
        write_mem(out_dir / "output_weight.mem", to_q8_8(out_weight.T, frac_w))
        write_mem(out_dir / "output_norm_weight.mem", to_q8_8(out_norm_w, frac_w))
        write_mem(out_dir / "output_norm_bias.mem", to_q8_8(out_norm_b, frac_w))

    attn_norm_w_all = []
    attn_norm_b_all = []
    qkv_w_all = []
    qkv_b_all = []
    attn_out_w_all = []
    attn_out_b_all = []
    ffn_norm_w_all = []
    ffn_norm_b_all = []
    ffn_up_w_all = []
    ffn_up_b_all = []
    ffn_dn_w_all = []
    ffn_dn_b_all = []

    for layer in range(n_layer):
        prefix = f"blk.{layer}."
        attn_norm_w = _tensor_data(_get_tensor(reader, prefix + "attn_norm.weight"))
        attn_norm_b = _tensor_data(_get_tensor(reader, prefix + "attn_norm.bias"))
        qkv_w = _tensor_data(_get_tensor(reader, prefix + "attn_qkv.weight"))
        qkv_b = _tensor_data(_get_tensor(reader, prefix + "attn_qkv.bias"))
        attn_out_w = _tensor_data(_get_tensor(reader, prefix + "attn_output.weight"))
        attn_out_b = _tensor_data(_get_tensor(reader, prefix + "attn_output.bias"))
        ffn_norm_w = _tensor_data(_get_tensor(reader, prefix + "ffn_norm.weight"))
        ffn_norm_b = _tensor_data(_get_tensor(reader, prefix + "ffn_norm.bias"))
        ffn_up_w = _tensor_data(_get_tensor(reader, prefix + "ffn_up.weight"))
        ffn_up_b = _tensor_data(_get_tensor(reader, prefix + "ffn_up.bias"))
        ffn_dn_w = _tensor_data(_get_tensor(reader, prefix + "ffn_down.weight"))
        ffn_dn_b = _tensor_data(_get_tensor(reader, prefix + "ffn_down.bias"))

        if qkv_w.shape != (n_embd, 3 * n_embd):
            raise RuntimeError(f"{prefix}attn_qkv.weight shape {qkv_w.shape}")
        if attn_out_w.shape != (n_embd, n_embd):
            raise RuntimeError(f"{prefix}attn_output.weight shape {attn_out_w.shape}")
        if ffn_up_w.shape != (n_embd, n_ff):
            raise RuntimeError(f"{prefix}ffn_up.weight shape {ffn_up_w.shape}")
        if ffn_dn_w.shape != (n_ff, n_embd):
            raise RuntimeError(f"{prefix}ffn_down.weight shape {ffn_dn_w.shape}")

        attn_norm_w_all.append(attn_norm_w)
        attn_norm_b_all.append(attn_norm_b)
        qkv_w_all.append(qkv_w)
        qkv_b_all.append(qkv_b)
        attn_out_w_all.append(attn_out_w)
        attn_out_b_all.append(attn_out_b)
        ffn_norm_w_all.append(ffn_norm_w)
        ffn_norm_b_all.append(ffn_norm_b)
        ffn_up_w_all.append(ffn_up_w)
        ffn_up_b_all.append(ffn_up_b)
        ffn_dn_w_all.append(ffn_dn_w)
        ffn_dn_b_all.append(ffn_dn_b)

    if fp16:
        write_mem_fp16(out_dir / "attn_norm_weight.mem", np.stack(attn_norm_w_all, axis=0))
        write_mem_fp16(out_dir / "attn_norm_bias.mem", np.stack(attn_norm_b_all, axis=0))
        write_mem_fp16(out_dir / "attn_qkv_weight.mem", np.stack(qkv_w_all, axis=0))
        write_mem_fp16(out_dir / "attn_qkv_bias.mem", np.stack(qkv_b_all, axis=0))
        write_mem_fp16(out_dir / "attn_output_weight.mem", np.stack(attn_out_w_all, axis=0))
        write_mem_fp16(out_dir / "attn_output_bias.mem", np.stack(attn_out_b_all, axis=0))
        write_mem_fp16(out_dir / "ffn_norm_weight.mem", np.stack(ffn_norm_w_all, axis=0))
        write_mem_fp16(out_dir / "ffn_norm_bias.mem", np.stack(ffn_norm_b_all, axis=0))
        write_mem_fp16(out_dir / "ffn_up_weight.mem", np.stack(ffn_up_w_all, axis=0))
        write_mem_fp16(out_dir / "ffn_up_bias.mem", np.stack(ffn_up_b_all, axis=0))
        write_mem_fp16(out_dir / "ffn_down_weight.mem", np.stack(ffn_dn_w_all, axis=0))
        write_mem_fp16(out_dir / "ffn_down_bias.mem", np.stack(ffn_dn_b_all, axis=0))
    else:
        write_mem(out_dir / "attn_norm_weight.mem", to_q8_8(np.stack(attn_norm_w_all, axis=0), frac_w))
        write_mem(out_dir / "attn_norm_bias.mem", to_q8_8(np.stack(attn_norm_b_all, axis=0), frac_w))
        write_mem(out_dir / "attn_qkv_weight.mem", to_q8_8(np.stack(qkv_w_all, axis=0), frac_w))
        write_mem(out_dir / "attn_qkv_bias.mem", to_q8_8(np.stack(qkv_b_all, axis=0), frac_w))
        write_mem(out_dir / "attn_output_weight.mem", to_q8_8(np.stack(attn_out_w_all, axis=0), frac_w))
        write_mem(out_dir / "attn_output_bias.mem", to_q8_8(np.stack(attn_out_b_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_norm_weight.mem", to_q8_8(np.stack(ffn_norm_w_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_norm_bias.mem", to_q8_8(np.stack(ffn_norm_b_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_up_weight.mem", to_q8_8(np.stack(ffn_up_w_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_up_bias.mem", to_q8_8(np.stack(ffn_up_b_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_down_weight.mem", to_q8_8(np.stack(ffn_dn_w_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_down_bias.mem", to_q8_8(np.stack(ffn_dn_b_all, axis=0), frac_w))

    mode = "FP16" if fp16 else "Q8.8"
    print(f"Exported GPT-2 {n_layer} layers, vocab {vocab}, d_model {n_embd} to {out_dir} ({mode})")


def export_llama(reader, out_dir, frac_w, fp16):
    n_layer = field_u32(reader, "llama.block_count")
    n_embd = field_u32(reader, "llama.embedding_length")
    n_ff = field_u32(reader, "llama.feed_forward_length")
    vocab = field_u32(reader, "llama.vocab_size")

    token_embd = _tensor_data(_get_tensor(reader, "token_embd.weight"))
    out_norm_w = _tensor_data(_get_tensor(reader, "output_norm.weight"))

    if token_embd.shape != (n_embd, vocab):
        raise RuntimeError(f"token_embd shape {token_embd.shape}")
    if out_norm_w.shape != (n_embd,):
        raise RuntimeError(f"output_norm.weight shape {out_norm_w.shape}")

    if fp16:
        write_mem_fp16(out_dir / "token_embd.mem", token_embd.T)
        write_mem_fp16(out_dir / "output_weight.mem", token_embd.T)
        write_mem_fp16(out_dir / "output_norm_weight.mem", out_norm_w)
    else:
        write_mem(out_dir / "token_embd.mem", to_q8_8(token_embd.T, frac_w))
        write_mem(out_dir / "output_weight.mem", to_q8_8(token_embd.T, frac_w))
        write_mem(out_dir / "output_norm_weight.mem", to_q8_8(out_norm_w, frac_w))

    attn_norm_w_all = []
    ffn_norm_w_all = []
    attn_q_w_all = []
    attn_k_w_all = []
    attn_v_w_all = []
    attn_out_w_all = []
    ffn_gate_w_all = []
    ffn_up_w_all = []
    ffn_dn_w_all = []

    for layer in range(n_layer):
        prefix = f"blk.{layer}."
        attn_norm_w = _tensor_data(_get_tensor(reader, prefix + "attn_norm.weight"))
        ffn_norm_w = _tensor_data(_get_tensor(reader, prefix + "ffn_norm.weight"))
        attn_q_w = _tensor_data(_get_tensor(reader, prefix + "attn_q.weight"))
        attn_k_w = _tensor_data(_get_tensor(reader, prefix + "attn_k.weight"))
        attn_v_w = _tensor_data(_get_tensor(reader, prefix + "attn_v.weight"))
        attn_out_w = _tensor_data(_get_tensor(reader, prefix + "attn_output.weight"))
        ffn_gate_w = _tensor_data(_get_tensor(reader, prefix + "ffn_gate.weight"))
        ffn_up_w = _tensor_data(_get_tensor(reader, prefix + "ffn_up.weight"))
        ffn_dn_w = _tensor_data(_get_tensor(reader, prefix + "ffn_down.weight"))

        if attn_q_w.shape != (n_embd, n_embd):
            raise RuntimeError(f"{prefix}attn_q.weight shape {attn_q_w.shape}")
        if attn_out_w.shape != (n_embd, n_embd):
            raise RuntimeError(f"{prefix}attn_output.weight shape {attn_out_w.shape}")
        if ffn_up_w.shape != (n_embd, n_ff):
            raise RuntimeError(f"{prefix}ffn_up.weight shape {ffn_up_w.shape}")
        if ffn_gate_w.shape != (n_embd, n_ff):
            raise RuntimeError(f"{prefix}ffn_gate.weight shape {ffn_gate_w.shape}")
        if ffn_dn_w.shape != (n_ff, n_embd):
            raise RuntimeError(f"{prefix}ffn_down.weight shape {ffn_dn_w.shape}")

        attn_norm_w_all.append(attn_norm_w)
        ffn_norm_w_all.append(ffn_norm_w)
        attn_q_w_all.append(attn_q_w)
        attn_k_w_all.append(attn_k_w)
        attn_v_w_all.append(attn_v_w)
        attn_out_w_all.append(attn_out_w)
        ffn_gate_w_all.append(ffn_gate_w)
        ffn_up_w_all.append(ffn_up_w)
        ffn_dn_w_all.append(ffn_dn_w)

    if fp16:
        write_mem_fp16(out_dir / "attn_norm_weight.mem", np.stack(attn_norm_w_all, axis=0))
        write_mem_fp16(out_dir / "ffn_norm_weight.mem", np.stack(ffn_norm_w_all, axis=0))
        write_mem_fp16(out_dir / "attn_q_weight.mem", np.stack(attn_q_w_all, axis=0))
        write_mem_fp16(out_dir / "attn_k_weight.mem", np.stack(attn_k_w_all, axis=0))
        write_mem_fp16(out_dir / "attn_v_weight.mem", np.stack(attn_v_w_all, axis=0))
        write_mem_fp16(out_dir / "attn_output_weight.mem", np.stack(attn_out_w_all, axis=0))
        write_mem_fp16(out_dir / "ffn_gate_weight.mem", np.stack(ffn_gate_w_all, axis=0))
        write_mem_fp16(out_dir / "ffn_up_weight.mem", np.stack(ffn_up_w_all, axis=0))
        write_mem_fp16(out_dir / "ffn_down_weight.mem", np.stack(ffn_dn_w_all, axis=0))
    else:
        write_mem(out_dir / "attn_norm_weight.mem", to_q8_8(np.stack(attn_norm_w_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_norm_weight.mem", to_q8_8(np.stack(ffn_norm_w_all, axis=0), frac_w))
        write_mem(out_dir / "attn_q_weight.mem", to_q8_8(np.stack(attn_q_w_all, axis=0), frac_w))
        write_mem(out_dir / "attn_k_weight.mem", to_q8_8(np.stack(attn_k_w_all, axis=0), frac_w))
        write_mem(out_dir / "attn_v_weight.mem", to_q8_8(np.stack(attn_v_w_all, axis=0), frac_w))
        write_mem(out_dir / "attn_output_weight.mem", to_q8_8(np.stack(attn_out_w_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_gate_weight.mem", to_q8_8(np.stack(ffn_gate_w_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_up_weight.mem", to_q8_8(np.stack(ffn_up_w_all, axis=0), frac_w))
        write_mem(out_dir / "ffn_down_weight.mem", to_q8_8(np.stack(ffn_dn_w_all, axis=0), frac_w))

    mode = "FP16" if fp16 else "Q8.8"
    print(f"Exported LLaMA {n_layer} layers, vocab {vocab}, d_model {n_embd} to {out_dir} ({mode})")


def export(model_path, out_dir, frac_w, fp16):
    reader = gguf.GGUFReader(str(model_path))

    arch = reader.fields.get("general.architecture")
    if arch is None:
        raise RuntimeError("Missing general.architecture in GGUF metadata")
    arch_name = str(arch.contents())

    out_dir.mkdir(parents=True, exist_ok=True)
    if arch_name == "gpt2":
        export_gpt2(reader, out_dir, frac_w, fp16)
    elif arch_name == "llama":
        export_llama(reader, out_dir, frac_w, fp16)
    else:
        raise RuntimeError(
            f"Unsupported architecture '{arch_name}'. This exporter supports GPT-2 and LLaMA GGUF only."
        )


def main():
    parser = argparse.ArgumentParser(description="Export full GGUF weights to Q8.8 mem files.")
    parser.add_argument("--model", required=True, help="Path to F16 GGUF model")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--frac-w", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", help="Write FP16 mem files instead of Q8.8")
    args = parser.parse_args()

    _require_gguf()
    export(Path(args.model), Path(args.out), args.frac_w, args.fp16)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
