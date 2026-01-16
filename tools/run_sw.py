#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def extract_generated_text(text):
    lines = text.splitlines()
    prompt_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("> "):
            prompt_idx = idx
    if prompt_idx is None:
        return text.strip("\n")
    gen_lines = []
    for line in lines[prompt_idx + 1 :]:
        if line.startswith("["):
            break
        gen_lines.append(line)
    return "\n".join(gen_lines).strip("\n")


def run_llama_generate(model_path, prompt_text, steps, do_sample, top_k, top_p, temperature, seed):
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
        str(steps),
        "--temp",
        str(temperature if do_sample else 0.0),
        "--top-k",
        str(top_k if do_sample else 1),
        "--top-p",
        str(top_p if do_sample else 1.0),
        "--repeat-penalty",
        "1.0",
        "--presence-penalty",
        "0",
        "--frequency-penalty",
        "0",
        "--seed",
        str(seed),
        "-ngl",
        "0",
    ]
    text = run(cmd, env=env)
    return extract_generated_text(text)


def parse_adder_prompt(prompt, ndigit):
    match = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*=\s*$", prompt)
    if not match:
        raise RuntimeError("Adder prompt must look like 'A+B=' with digits only")
    a = int(match.group(1))
    b = int(match.group(2))
    a_str = f"%0{ndigit}d" % a
    b_str = f"%0{ndigit}d" % b
    if len(a_str) != ndigit or len(b_str) != ndigit:
        raise RuntimeError("Parsed numbers do not match ndigit")
    return a, b


def run_sw_adder_inference(config_path, prompt):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ndigit = int(cfg["data"]["ndigit"])
    a, b = parse_adder_prompt(prompt, ndigit)
    return a + b


def run_sw_inference(
    gguf_path,
    prompt,
    steps,
    do_sample,
    top_k,
    top_p,
    temperature,
    seed,
    interactive,
):
    prompt_text = prompt or ""
    if not interactive:
        output = run_llama_generate(
            gguf_path,
            prompt_text,
            steps,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
        )
        print(output, end="")
        return

    print("Enter prompt lines. Ctrl+D to exit.", file=sys.stderr)
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        prompt_text += line
        output = run_llama_generate(
            gguf_path,
            prompt_text,
            steps,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
        )
        prompt_text += output
        print(output, end="", flush=True)


if __name__ == "__main__":
    print("error: run_sw.py is a module; use chat.py", file=sys.stderr)
    sys.exit(1)
