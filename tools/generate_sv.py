#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "llm-models" / "tinystories-gpt-0.1-3m.fp16.gguf"
DEFAULT_WEIGHTS = ROOT / "llm-models" / "weights_tinystories_fp16"
DEFAULT_PROMPT = "<|start_story|>Once upon a time, "


def run(cmd):
    subprocess.run(cmd, check=True, text=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a sample using SV/Verilator (HW path)."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to GGUF model")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS), help="Exported weights dir")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--steps", type=int, default=200, help="Tokens to generate")
    parser.add_argument("--seed", type=int, default=1, help="Sampling seed")
    parser.add_argument("--top-k", type=int, default=40, help="Sampling top-k")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--gpt2-fp16", action="store_true", help="Use FP16 GPT-2 SV model")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "chat.py"),
        "--model",
        str(args.model),
        "--weights-dir",
        str(args.weights_dir),
        "--prompt",
        args.prompt,
        "--steps",
        str(args.steps),
        "--do-sample",
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
    ]
    if args.gpt2_fp16:
        cmd.append("--gpt2-fp16")

    run(cmd)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
