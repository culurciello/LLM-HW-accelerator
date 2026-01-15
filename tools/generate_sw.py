#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "llm-models" / "tinystories-gpt-0.1-3m.fp16.gguf"
DEFAULT_PROMPT = "<|start_story|>Once upon a time, "


def run(cmd, env=None):
    subprocess.run(cmd, check=True, text=True, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a sample using llama.cpp (SW reference)."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to GGUF model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--n-predict", type=int, default=200, help="Tokens to generate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--top-k", type=int, default=40, help="Sampling top-k")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise RuntimeError(f"Missing model: {model_path}")

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
        args.prompt,
        "-n",
        str(args.n_predict),
        "--temp",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--repeat-penalty",
        "1.0",
        "--presence-penalty",
        "0",
        "--frequency-penalty",
        "0",
        "--seed",
        str(args.seed),
        "-ngl",
        "0",
    ]
    run(cmd, env=env)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
