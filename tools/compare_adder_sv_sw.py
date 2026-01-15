#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "llm-models" / "adder" / "model.pt"
DEFAULT_CONFIG = ROOT / "llm-models" / "adder" / "config.json"
DEFAULT_WEIGHTS_DIR = ROOT / "llm-models" / "adder" / "weights_sv"
DEFAULT_PROMPT_FILE = ROOT / "tb" / "prompt_ids.mem"
DEFAULT_PROMPT = "3+3="


def _load_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def _parse_prompt(prompt, ndigit):
    m = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*=\s*$", prompt)
    if not m:
        raise RuntimeError("Prompt must look like 'A+B=' with digits only")
    a = int(m.group(1))
    b = int(m.group(2))
    a_str = f"%0{ndigit}d" % a
    b_str = f"%0{ndigit}d" % b
    if len(a_str) != ndigit or len(b_str) != ndigit:
        raise RuntimeError("Parsed numbers do not match ndigit")
    digits = [int(ch) for ch in (a_str + b_str)]
    return digits, a, b


def _run(cmd, cwd=ROOT, env=None):
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


def _run_sv(weights_dir, prompt_file, prompt_digits, steps):
    run_script = ROOT / "tb" / "run_adder_infer.sh"
    ids = list(prompt_digits)
    out_digits = []
    for _ in range(steps):
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        with prompt_file.open("w", encoding="ascii") as f:
            for tid in ids:
                f.write(f"{tid:08x}\n")
        cmd = [
            str(run_script),
            f"+weights_dir={weights_dir}",
            f"+prompt_ids={prompt_file}",
            f"+prompt_len={len(ids)}",
        ]
        out = _run(cmd)
        next_id = None
        for line in out.splitlines():
            if line.startswith("NEXT_TOKEN_ID="):
                next_id = int(line.split("=")[1])
                break
        if next_id is None:
            raise RuntimeError("Missing NEXT_TOKEN_ID from SV run")
        out_digits.append(next_id)
        ids.append(next_id)
    return out_digits


def _run_sw(checkpoint, config_path, prompt_digits, steps):
    sys.path.insert(0, str(ROOT.parent / "minGPT-master"))
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing Python package 'torch'.") from exc
    from mingpt.model import GPT

    config = _load_config(config_path)
    ndigit = int(config["data"]["ndigit"])
    model_cfg = config["model"]

    cfg = GPT.get_default_config()
    if model_cfg.get("model_type"):
        cfg.model_type = model_cfg["model_type"]
    else:
        cfg.model_type = None
        cfg.n_layer = model_cfg.get("n_layer")
        cfg.n_head = model_cfg.get("n_head")
        cfg.n_embd = model_cfg.get("n_embd")
    cfg.vocab_size = 10
    cfg.block_size = 3 * ndigit

    model = GPT(cfg)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    x = torch.tensor([prompt_digits], dtype=torch.long)
    y = model.generate(x, steps, do_sample=False)
    out_digits = y[0, -steps:].tolist()
    return out_digits


def _digits_to_sum(digits):
    normal = list(reversed(digits))
    return int("".join(str(d) for d in normal)), normal


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare minGPT adder model outputs between SW and SV."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    ndigit = int(cfg["data"]["ndigit"])
    prompt_digits, a, b = _parse_prompt(args.prompt, ndigit)

    steps = ndigit + 1
    sv_digits = _run_sv(Path(args.weights_dir), Path(args.prompt_file), prompt_digits, steps)
    sw_digits = _run_sw(Path(args.checkpoint), Path(args.config), prompt_digits, steps)

    sv_sum, sv_normal = _digits_to_sum(sv_digits)
    sw_sum, sw_normal = _digits_to_sum(sw_digits)

    print(f"prompt: {args.prompt!r}")
    print(f"ndigit: {ndigit}")
    print(f"inputs: {a} + {b}")
    print(f"prompt_digits: {prompt_digits}")
    print(f"sv_out_digits_reversed: {sv_digits}")
    print(f"sw_out_digits_reversed: {sw_digits}")
    print(f"sv_out_digits_normal: {sv_normal}")
    print(f"sw_out_digits_normal: {sw_normal}")
    print(f"sv_sum: {sv_sum}")
    print(f"sw_sum: {sw_sum}")
    print(f"sv_vs_sw: {'MATCH' if sv_digits == sw_digits else 'MISMATCH'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
