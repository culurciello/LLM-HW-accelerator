#!/usr/bin/env python3
import ast
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def gguf_architecture(model_path):
    if gguf is None:
        return None
    reader = gguf.GGUFReader(str(model_path))
    field = reader.fields.get("general.architecture")
    if field is None:
        return None
    return str(field.contents()).strip()


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
    next_id = None
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
    if next_id is None:
        raise RuntimeError("Missing NEXT_TOKEN_ID from Verilator")
    return next_id, (topk if topk else None)


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


def run_sv_llm(
    model_path,
    run_script,
    weights_dir,
    ids,
    steps,
    vocab,
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
        out_text.append(token_text)
    return "".join(out_text)


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
    digits = [int(ch) for ch in (a_str + b_str)]
    return digits


def run_sv_adder(weights_dir, prompt_digits, steps):
    run_script = ROOT / "tb" / "run_adder_infer.sh"
    ids = list(prompt_digits)
    out_digits = []
    for _ in range(steps):
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


def digits_to_sum(digits):
    normal = list(reversed(digits))
    return int("".join(str(d) for d in normal))


def resolve_sv_impl(gguf_path, sv_impl):
    if sv_impl == "auto":
        arch = gguf_architecture(gguf_path)
        if arch is None:
            raise RuntimeError("Unable to detect GGUF architecture; use --sv-impl")
        arch_lower = arch.lower()
        if arch_lower == "llama":
            sv_impl = "llama"
        elif "gpt" in arch_lower:
            sv_impl = "gpt2"
        else:
            raise RuntimeError(f"Unsupported GGUF architecture: {arch}")

    if sv_impl == "llama":
        return ROOT / "tb" / "run_llama_infer.sh", "fp16"
    if sv_impl == "gpt2":
        return ROOT / "tb" / "run_gpt2_infer.sh", "fp16"
    if sv_impl == "gpt2-fixed":
        return ROOT / "tb" / "run_infer.sh", "q"
    raise RuntimeError(f"Unsupported SV impl: {sv_impl}")


def run_hw_llm_inference(
    gguf_path,
    weights_dir,
    prompt,
    steps,
    do_sample,
    top_k,
    top_p,
    temperature,
    seed,
    frac_w,
    sv_impl,
    interactive,
):
    run_script, score_mode = resolve_sv_impl(gguf_path, sv_impl)
    vocab = load_vocab(gguf_path)
    ids = []
    prompt_text = ""

    if not interactive:
        prompt_text = prompt
        ids = tokenize(gguf_path, prompt)
        output = run_sv_llm(
            gguf_path,
            run_script,
            weights_dir,
            ids,
            steps,
            vocab,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            score_mode,
            frac_w,
        )
        print(output, end="")
        return

    print("Enter prompt lines. Ctrl+D to exit.", file=sys.stderr)
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        prompt_text += line
        new_ids = tokenize(gguf_path, line)
        if new_ids:
            ids.extend(new_ids)
        output = run_sv_llm(
            gguf_path,
            run_script,
            weights_dir,
            ids,
            steps,
            vocab,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            score_mode,
            frac_w,
        )
        prompt_text += output
        print(output, end="", flush=True)


def run_hw_adder_inference(weights_dir, config_path, prompt, steps):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ndigit = int(cfg["data"]["ndigit"])
    prompt_digits = parse_adder_prompt(prompt, ndigit)
    total_steps = steps if steps is not None else (ndigit + 1)
    out_digits = run_sv_adder(weights_dir, prompt_digits, total_steps)
    return digits_to_sum(out_digits)


if __name__ == "__main__":
    print("error: run_hw.py is a module; use chat.py", file=sys.stderr)
    sys.exit(1)
