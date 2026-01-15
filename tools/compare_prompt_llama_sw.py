#!/usr/bin/env python3
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_F16 = ROOT / "llm-models" / "SmolLM2-135M-Instruct-f16.gguf"
DEFAULT_WEIGHTS_DIR = ROOT / "llm-models" / "weights_full"
DEFAULT_PROMPT_FILE = ROOT / "tb" / "prompt_ids.mem"
DEFAULT_PROMPT = "3+3="

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


def run_verilator(weights_dir, prompt_file, prompt_len):
    run_script = ROOT / "tb" / "run_llama_infer.sh"
    cmd = [
        str(run_script),
        f"+weights_dir={weights_dir}",
        f"+prompt_ids={prompt_file}",
        f"+prompt_len={prompt_len}",
    ]
    out = run(cmd)
    for line in out.splitlines():
        if line.startswith("NEXT_TOKEN_ID="):
            return int(line.split("=")[1])
    raise RuntimeError("Missing NEXT_TOKEN_ID from Verilator")


def run_llama_cli(model_f16, prompt):
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

    parser = argparse.ArgumentParser(
        description="Compare SV vs llama.cpp on the prompt '3+3=' (LLaMA path)."
    )
    parser.add_argument("--model-f16", default=str(DEFAULT_MODEL_F16))
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    args = parser.parse_args()

    model_f16 = Path(args.model_f16)
    weights_dir = Path(args.weights_dir)
    prompt_file = Path(args.prompt_file)
    prompt = args.prompt

    ensure_f16(model_f16)
    if not weights_dir.exists():
        raise RuntimeError(f"Missing weights dir: {weights_dir}")

    ids = tokenize_prompt(model_f16, prompt, prompt_file)
    sv_token = run_verilator(weights_dir, prompt_file, len(ids))
    llama_token, llama_text = run_llama_cli(model_f16, prompt)

    print(f"prompt: {prompt!r}")
    print(f"prompt_ids: {ids}")
    print(f"sv_next_token_id: {sv_token}")
    print(f"llama_next_token_id: {llama_token}")
    print(f"llama_next_token_text: {llama_text}")
    print(f"sv_vs_llama: {'MATCH' if sv_token == llama_token else 'MISMATCH'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
