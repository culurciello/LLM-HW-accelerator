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
    if hasattr(reader, "tensors"):
        for t in reader.tensors:
            if getattr(t, "name", None) == name:
                return t
    if hasattr(reader, "get_tensor"):
        return reader.get_tensor(name)
    raise KeyError(name)


def _tensor_data(tensor):
    if hasattr(tensor, "data"):
        data = tensor.data
    elif hasattr(tensor, "get_data"):
        data = tensor.get_data()
    else:
        raise AttributeError("tensor has no data accessor")
    arr = np.array(data, dtype=np.float32)
    shape = getattr(tensor, "shape", None)
    if shape is not None:
        try:
            arr = arr.reshape(shape)
        except Exception:
            pass
    return arr


def to_q8_8(arr, frac_w):
    scale = 1 << frac_w
    scaled = np.round(arr * scale).astype(np.int32)
    clipped = np.clip(scaled, -32768, 32767).astype(np.int16)
    return clipped


def write_mem(path, arr):
    with path.open("w", encoding="ascii") as f:
        for val in arr.flatten():
            f.write(f"{int(val) & 0xFFFF:04x}\n")

def _require_shape(mat, rows, cols, name):
    if mat.shape[0] < rows or mat.shape[1] < cols:
        raise RuntimeError(
            f"{name} has shape {mat.shape}, need at least ({rows}, {cols})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF weights to accelerator mem files."
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--block", type=int, default=0, help="Transformer block index")
    parser.add_argument("--d-model", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=8)
    parser.add_argument("--frac-w", type=int, default=8)
    args = parser.parse_args()

    _require_gguf()

    model_path = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = gguf.GGUFReader(str(model_path))

    blk = args.block
    qkv_name = f"blk.{blk}.attn_qkv.weight"
    o_name = f"blk.{blk}.attn_output.weight"
    up_name = f"blk.{blk}.ffn_up.weight"
    down_name = f"blk.{blk}.ffn_down.weight"

    qkv = _tensor_data(_get_tensor(reader, qkv_name))
    o = _tensor_data(_get_tensor(reader, o_name))
    up = _tensor_data(_get_tensor(reader, up_name))
    down = _tensor_data(_get_tensor(reader, down_name))

    if qkv.ndim != 2 or o.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise RuntimeError("Expected 2D tensors; check GGUF reader output")

    # Split QKV and down-project to match accelerator dimensions.
    if qkv.shape[1] % 3 != 0:
        raise RuntimeError("QKV weight second dimension is not divisible by 3")
    split = qkv.shape[1] // 3
    q = qkv[:, :split]
    k = qkv[:, split:2 * split]
    v = qkv[:, 2 * split:3 * split]

    d_model = args.d_model
    d_ff = args.d_ff

    _require_shape(q, d_model, d_model, "WQ")
    _require_shape(k, d_model, d_model, "WK")
    _require_shape(v, d_model, d_model, "WV")
    _require_shape(o, d_model, d_model, "WO")
    _require_shape(up, d_model, d_ff, "W1")
    _require_shape(down, d_ff, d_model, "W2")

    wq = q[:d_model, :d_model]
    wk = k[:d_model, :d_model]
    wv = v[:d_model, :d_model]
    wo = o[:d_model, :d_model]
    w1 = up[:d_model, :d_ff]
    w2 = down[:d_ff, :d_model]

    write_mem(out_dir / "wq.mem", to_q8_8(wq, args.frac_w))
    write_mem(out_dir / "wk.mem", to_q8_8(wk, args.frac_w))
    write_mem(out_dir / "wv.mem", to_q8_8(wv, args.frac_w))
    write_mem(out_dir / "wo.mem", to_q8_8(wo, args.frac_w))
    write_mem(out_dir / "w1.mem", to_q8_8(w1, args.frac_w))
    write_mem(out_dir / "w2.mem", to_q8_8(w2, args.frac_w))

    print(f"Wrote mem files to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
