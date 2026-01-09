#!/usr/bin/env python3
import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "llm-models" / "SmolLM2-135M-Instruct-f16.gguf"
DEFAULT_WEIGHTS = ROOT / "llm-models" / "weights_full"
PROMPT_MEM = ROOT / "tb" / "prompt_ids.mem"

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


def run_verilator(run_script, weights_dir, ids):
    PROMPT_MEM.parent.mkdir(parents=True, exist_ok=True)
    with PROMPT_MEM.open("w", encoding="ascii") as f:
        for tid in ids:
            f.write(f"{tid:08x}\n")
    cmd = [
        str(run_script),
        f"+weights_dir={weights_dir}",
        f"+prompt_ids={PROMPT_MEM}",
        f"+prompt_len={len(ids)}",
    ]
    out = run(cmd)
    for line in out.splitlines():
        if line.startswith("NEXT_TOKEN_ID="):
            return int(line.split("=")[1])
    raise RuntimeError("Missing NEXT_TOKEN_ID from Verilator")


def load_vocab(model_path):
    if gguf is None:
        return None
    reader = gguf.GGUFReader(str(model_path))
    tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    return [str(tok) for tok in tokens]


def decode_tokens(tokens, vocab):
    if vocab is None:
        return ""
    pieces = []
    for tid in tokens:
        if 0 <= tid < len(vocab):
            pieces.append(vocab[tid])
    text = "".join(pieces)
    text = text.replace("\u0120", " ")
    text = text.replace("\u010a", "\n")
    return text


def run_llama_next_token(model_path, prompt_text):
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
        prompt_text,
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
    return token_text


def run_generation(model_path, run_script, weights_dir, ids, steps, vocab, compare_cpu, prompt_text):
    out_text = []
    for _ in range(steps):
        next_id = run_verilator(run_script, weights_dir, ids)
        ids.append(next_id)
        token_text = decode_tokens([next_id], vocab)
        if token_text == "" and vocab is None:
            token_text = f"<{next_id}>"
        if compare_cpu:
            cpu_token_text = run_llama_next_token(model_path, prompt_text)
            if cpu_token_text == "":
                cpu_token_text = "<eos>"
            print(f"HDL: {token_text} | CPU: {cpu_token_text}")
            prompt_text += cpu_token_text
        else:
            out_text.append(token_text)
        prompt_text += token_text
    return "".join(out_text), prompt_text


def main():
    parser = argparse.ArgumentParser(description="Chat with the HDL model via Verilator.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to F16 GGUF model")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS), help="Path to exported weights")
    parser.add_argument("--steps", type=int, default=16, help="Tokens to generate per prompt")
    parser.add_argument("--prompt", default=None, help="Prompt text (single-shot mode)")
    parser.add_argument("--compare-cpu", action="store_true", help="Print HDL vs CPU tokens per step")
    args = parser.parse_args()

    model_path = Path(args.model)
    weights_dir = Path(args.weights_dir)
    if not model_path.exists():
        raise RuntimeError(f"Missing model: {model_path}")
    if not weights_dir.exists():
        raise RuntimeError(f"Missing weights dir: {weights_dir}")

    if not weights_dir.exists():
        raise RuntimeError(f"Missing weights dir: {weights_dir}")

    arch_name = None
    if gguf is not None:
        reader = gguf.GGUFReader(str(model_path))
        arch = reader.fields.get("general.architecture")
        if arch is not None:
            arch_name = str(arch.contents())
    run_script = ROOT / "tb" / ("run_llama_infer.sh" if arch_name == "llama" else "run_infer.sh")

    vocab = load_vocab(model_path)
    ids = []
    prompt_text = ""

    if args.prompt is not None:
        prompt_text = args.prompt
        ids = tokenize(model_path, args.prompt)
        output, _ = run_generation(
            model_path,
            run_script,
            weights_dir,
            ids,
            args.steps,
            vocab,
            args.compare_cpu,
            prompt_text,
        )
        if not args.compare_cpu:
            print(output, end="")
        return

    print("Enter prompt lines. Ctrl+D to exit.", file=sys.stderr)
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        prompt_text += line
        new_ids = tokenize(model_path, line)
        if new_ids:
            ids.extend(new_ids)
        output, prompt_text = run_generation(
            model_path,
            run_script,
            weights_dir,
            ids,
            args.steps,
            vocab,
            args.compare_cpu,
            prompt_text,
        )
        if not args.compare_cpu:
            print(output, end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
