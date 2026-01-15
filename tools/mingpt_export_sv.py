#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:
    torch = None
    _torch_import_error = exc


MODEL_TYPE_MAP = {
    "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
}


def _require_torch():
    if torch is None:
        raise RuntimeError("Missing Python package 'torch'.") from _torch_import_error


def write_mem_fp16(path, arr):
    fp16 = np.asarray(arr, dtype=np.float16)
    with path.open("w", encoding="ascii") as f:
        for val in fp16.flatten(order="C").view(np.uint16):
            f.write(f"{int(val) & 0xFFFF:04x}\n")


def parse_config(config_path):
    if config_path is None:
        return {}
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = data.get("model", {})
    n_layer = model.get("n_layer")
    n_head = model.get("n_head")
    n_embd = model.get("n_embd")
    model_type = model.get("model_type")
    if model_type in MODEL_TYPE_MAP:
        defaults = MODEL_TYPE_MAP[model_type]
        if n_layer is None:
            n_layer = defaults["n_layer"]
        if n_head is None:
            n_head = defaults["n_head"]
        if n_embd is None:
            n_embd = defaults["n_embd"]
    return {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export a minGPT checkpoint to SV .mem files (FP16)."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", help="Optional config.json for n_head")
    parser.add_argument("--n-head", type=int, help="Override number of heads")
    args = parser.parse_args()

    _require_torch()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(ckpt_path, map_location="cpu")

    def get_tensor(name):
        if name not in state:
            raise KeyError(f"Missing tensor {name}")
        return state[name].detach().cpu().numpy()

    wte = get_tensor("transformer.wte.weight")
    wpe = get_tensor("transformer.wpe.weight")
    vocab, d_model = wte.shape
    max_ctx, d_model_wpe = wpe.shape
    if d_model_wpe != d_model:
        raise RuntimeError("wpe embedding size mismatch")

    layer = 0
    while f"transformer.h.{layer}.ln_1.weight" in state:
        layer += 1
    n_layer = layer
    if n_layer == 0:
        raise RuntimeError("Could not find any transformer blocks")

    c_fc = get_tensor("transformer.h.0.mlp.c_fc.weight")
    d_ff, d_model_fc = c_fc.shape
    if d_model_fc != d_model:
        raise RuntimeError("c_fc input size mismatch")

    config = parse_config(args.config)
    n_head = args.n_head if args.n_head is not None else config.get("n_head")
    if n_head is None:
        raise RuntimeError("n_head is required; pass --n-head or --config")

    if config.get("n_layer") is not None and config["n_layer"] != n_layer:
        raise RuntimeError("config n_layer does not match checkpoint")
    if config.get("n_embd") is not None and config["n_embd"] != d_model:
        raise RuntimeError("config n_embd does not match checkpoint")

    write_mem_fp16(out_dir / "token_embd.mem", wte)
    write_mem_fp16(out_dir / "pos_embd.mem", wpe)
    write_mem_fp16(out_dir / "output_weight.mem", get_tensor("lm_head.weight"))
    write_mem_fp16(out_dir / "output_norm_weight.mem", get_tensor("transformer.ln_f.weight"))
    write_mem_fp16(out_dir / "output_norm_bias.mem", get_tensor("transformer.ln_f.bias"))

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

    for idx in range(n_layer):
        prefix = f"transformer.h.{idx}."
        attn_norm_w_all.append(get_tensor(prefix + "ln_1.weight"))
        attn_norm_b_all.append(get_tensor(prefix + "ln_1.bias"))

        qkv_w = get_tensor(prefix + "attn.c_attn.weight").T
        qkv_b = get_tensor(prefix + "attn.c_attn.bias")
        attn_out_w = get_tensor(prefix + "attn.c_proj.weight").T
        attn_out_b = get_tensor(prefix + "attn.c_proj.bias")

        ffn_norm_w_all.append(get_tensor(prefix + "ln_2.weight"))
        ffn_norm_b_all.append(get_tensor(prefix + "ln_2.bias"))
        ffn_up_w = get_tensor(prefix + "mlp.c_fc.weight").T
        ffn_up_b = get_tensor(prefix + "mlp.c_fc.bias")
        ffn_dn_w = get_tensor(prefix + "mlp.c_proj.weight").T
        ffn_dn_b = get_tensor(prefix + "mlp.c_proj.bias")

        qkv_w_all.append(qkv_w)
        qkv_b_all.append(qkv_b)
        attn_out_w_all.append(attn_out_w)
        attn_out_b_all.append(attn_out_b)
        ffn_up_w_all.append(ffn_up_w)
        ffn_up_b_all.append(ffn_up_b)
        ffn_dn_w_all.append(ffn_dn_w)
        ffn_dn_b_all.append(ffn_dn_b)

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

    params = {
        "n_layer": n_layer,
        "n_head": n_head,
        "d_model": d_model,
        "d_ff": d_ff,
        "max_ctx": max_ctx,
        "vocab": vocab,
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Exported minGPT checkpoint to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
