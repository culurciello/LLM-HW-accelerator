#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "llm-models" / "tinystories-gpt-0.1-3m.fp16.gguf"
DEFAULT_PROMPT = "<|start_story|>Once upon a time, "
DEFAULT_N_PREDICT = 200


def run(cmd, env=None):
    subprocess.run(cmd, check=True, text=True, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Generate TinyStories samples with the recommended prompt and sampling config."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to TinyStories GGUF model",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text (defaults to the TinyStories template)",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=DEFAULT_N_PREDICT,
        help="Number of tokens to generate",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
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
        "0.6",
        "--top-k",
        "40",
        "--top-p",
        "0.9",
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
