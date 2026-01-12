#!/usr/bin/env python3
import argparse
import ast
import os
import random
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


def run_verilator(run_script, weights_dir, ids, dump_topk=False):
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
    if dump_topk:
        cmd.append("+dump_topk=1")
    out = run(cmd)
    topk = []
    for line in out.splitlines():
        if line.startswith("NEXT_TOKEN_ID="):
            next_id = int(line.split("=")[1])
        elif line.startswith("TOPK["):
            parts = line.split()
            tid = int(parts[1].split("=")[1])
            score = None
            for part in parts[2:]:
                if part.startswith("score_fp16="):
                    score = part.split("=")[1]
                    try:
                        score = int(score, 16)
                    except ValueError:
                        score = None
                    break
                if part.startswith("score_q=") or part.startswith("score="):
                    score = int(part.split("=")[1])
                    break
            topk.append((tid, score))
    if "next_id" not in locals():
        raise RuntimeError("Missing NEXT_TOKEN_ID from Verilator")
    return next_id, (topk if topk else None)


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


def run_llama_next_token(model_path, prompt_text, do_sample, top_k, top_p, temperature, seed):
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


def fp16_to_float(val):
    sign = (val >> 15) & 0x1
    exp = (val >> 10) & 0x1F
    mant = val & 0x3FF
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        return ((-1.0) ** sign) * (mant / 1024.0) * (2.0 ** (-14))
    if exp == 0x1F:
        return float("-inf") if sign else float("inf")
    return ((-1.0) ** sign) * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))


def sample_from_topk(topk, temperature, top_p, top_k, rng):
    if top_k > 0 and len(topk) > top_k:
        topk = topk[:top_k]
    logits = [score for _, score in topk]
    if temperature <= 0:
        return topk[0][0]
    max_logit = max(logits)
    exp_vals = [pow(2.718281828, (l - max_logit) / temperature) for l in logits]
    total = sum(exp_vals)
    probs = [v / total for v in exp_vals]
    cumulative = 0.0
    cutoff = len(probs)
    for i, p in enumerate(probs):
        cumulative += p
        if cumulative >= top_p:
            cutoff = i + 1
            break
    if cutoff < len(probs):
        probs = probs[:cutoff]
        topk = topk[:cutoff]
        total = sum(probs)
        probs = [p / total for p in probs]
    r = rng.random()
    acc = 0.0
    for (tid, _), p in zip(topk, probs):
        acc += p
        if r <= acc:
            return tid
    return topk[-1][0]


def run_generation(
    model_path,
    run_script,
    weights_dir,
    ids,
    steps,
    vocab,
    compare_cpu,
    prompt_text,
    do_sample,
    top_k,
    top_p,
    temperature,
    seed,
    score_mode,
    frac_w,
):
    out_text = []
    rng = random.Random(seed)
    for _ in range(steps):
        next_id, topk = run_verilator(run_script, weights_dir, ids, dump_topk=do_sample)
        if do_sample:
            if not topk:
                raise RuntimeError("Missing TOPK output for sampling")
            scored = []
            for tid, score in topk:
                if score is None:
                    continue
                if score_mode == "fp16":
                    scored.append((tid, fp16_to_float(score)))
                else:
                    scored.append((tid, score / float(1 << frac_w)))
            if not scored:
                raise RuntimeError("TOPK scores missing or invalid")
            next_id = sample_from_topk(scored, temperature, top_p, top_k, rng)
        ids.append(next_id)
        token_text = decode_tokens([next_id], vocab)
        if token_text == "" and vocab is None:
            token_text = f"<{next_id}>"
        if compare_cpu:
            cpu_token_text = run_llama_next_token(
                model_path,
                prompt_text,
                do_sample,
                top_k,
                top_p,
                temperature,
                seed,
            )
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
    parser.add_argument("--do-sample", action="store_true", help="Enable top-k/top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Sampling top-k")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=1, help="Sampling seed")
    parser.add_argument("--frac-w", type=int, default=8, help="Fixed-point frac width for GPT2 scores")
    parser.add_argument("--gpt2-fp16", action="store_true", help="Use FP16 GPT-2 SV model")
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
    is_llama = arch_name == "llama"
    if is_llama:
        run_script = ROOT / "tb" / "run_llama_infer.sh"
        score_mode = "fp16"
    elif args.gpt2_fp16:
        run_script = ROOT / "tb" / "run_gpt2_infer.sh"
        score_mode = "fp16"
    else:
        run_script = ROOT / "tb" / "run_infer.sh"
        score_mode = "q"

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
            args.do_sample,
            args.top_k,
            args.top_p,
            args.temperature,
            args.seed,
            score_mode,
            args.frac_w,
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
            args.do_sample,
            args.top_k,
            args.top_p,
            args.temperature,
            args.seed,
            score_mode,
            args.frac_w,
        )
        if not args.compare_cpu:
            print(output, end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
