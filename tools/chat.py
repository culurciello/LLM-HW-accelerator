#!/usr/bin/env python3
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import run_hw
import run_sw

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "llm-models"


@dataclass
class ModelInfo:
    name: str
    path: Path
    gguf_files: list
    weights_dirs: list
    config_path: Path | None
    checkpoint_path: Path | None


def discover_models(models_root=MODELS_ROOT):
    models = {}
    if not models_root.exists():
        return models
    for path in sorted(models_root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        gguf_files = sorted(path.glob("*.gguf"))
        weights_dirs = sorted(
            [p for p in path.iterdir() if p.is_dir() and p.name.startswith("weights")]
        )
        config_path = path / "config.json"
        checkpoint_path = path / "model.pt"
        if gguf_files or weights_dirs or config_path.exists() or checkpoint_path.exists():
            models[path.name] = ModelInfo(
                name=path.name,
                path=path,
                gguf_files=gguf_files,
                weights_dirs=weights_dirs,
                config_path=config_path if config_path.exists() else None,
                checkpoint_path=checkpoint_path if checkpoint_path.exists() else None,
            )
    return models


def describe_models(models):
    if not models:
        print("No models found under llm-models/", file=sys.stderr)
        return
    for name, info in models.items():
        ggufs = ", ".join(p.name for p in info.gguf_files) or "-"
        weights = ", ".join(p.name for p in info.weights_dirs) or "-"
        extras = []
        if info.config_path:
            extras.append("config.json")
        if info.checkpoint_path:
            extras.append("model.pt")
        extras_str = ", ".join(extras) or "-"
        print(f"{name}: gguf=[{ggufs}] weights=[{weights}] extras=[{extras_str}]")


def resolve_model(arg, models):
    if arg is None:
        if len(models) == 1:
            return next(iter(models.values()))
        raise RuntimeError("Multiple models found; use --model or --list-models.")

    path = Path(arg)
    if path.exists():
        if path.is_file():
            model_path = path.parent
        else:
            model_path = path
        return build_model_info(model_path)

    if arg in models:
        return models[arg]

    raise RuntimeError(f"Unknown model: {arg}")


def build_model_info(path):
    gguf_files = sorted(path.glob("*.gguf"))
    weights_dirs = sorted([p for p in path.iterdir() if p.is_dir() and p.name.startswith("weights")])
    config_path = path / "config.json"
    checkpoint_path = path / "model.pt"
    return ModelInfo(
        name=path.name,
        path=path,
        gguf_files=gguf_files,
        weights_dirs=weights_dirs,
        config_path=config_path if config_path.exists() else None,
        checkpoint_path=checkpoint_path if checkpoint_path.exists() else None,
    )


def select_gguf(info, gguf_arg):
    if gguf_arg:
        gguf_path = Path(gguf_arg)
        if not gguf_path.is_absolute():
            gguf_path = info.path / gguf_path
        if not gguf_path.exists():
            raise RuntimeError(f"Missing GGUF model: {gguf_path}")
        return gguf_path

    if len(info.gguf_files) == 1:
        return info.gguf_files[0]
    if not info.gguf_files:
        return None
    raise RuntimeError(
        "Multiple GGUF files found; use --gguf to pick one: "
        + ", ".join(p.name for p in info.gguf_files)
    )


def select_weights(info, weights_arg):
    if weights_arg:
        weights_path = Path(weights_arg)
        if not weights_path.is_absolute():
            weights_path = info.path / weights_path
        if not weights_path.exists():
            raise RuntimeError(f"Missing weights dir: {weights_path}")
        return weights_path

    if not info.weights_dirs:
        return None

    def score(path):
        name = path.name.lower()
        if "fp16" in name:
            return 0
        if "sv" in name:
            return 1
        return 2

    return sorted(info.weights_dirs, key=score)[0]


def main():
    parser = argparse.ArgumentParser(description="Run LLM inference in SW (llama) or SV.")
    parser.add_argument("--list-models", action="store_true", help="List models under llm-models/")
    parser.add_argument("--model", help="Model name in llm-models/ or a path to a model dir/gguf")
    parser.add_argument("--gguf", help="GGUF filename or path within the model dir")
    parser.add_argument("--weights", help="Weights dir name or path within the model dir")
    parser.add_argument(
        "--backend",
        choices=["sv", "sw"],
        default="sv",
        help="Run SystemVerilog (sv) or llama.cpp (sw) inference",
    )
    parser.add_argument(
        "--sv-impl",
        choices=["auto", "llama", "gpt2", "gpt2-fixed", "adder"],
        default="auto",
        help="SV inference variant (auto uses GGUF metadata when available)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Tokens to generate per prompt")
    parser.add_argument("--prompt", default=None, help="Prompt text (single-shot mode)")
    parser.add_argument("--do-sample", action="store_true", help="Enable top-k/top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Sampling top-k")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=1, help="Sampling seed")
    parser.add_argument("--frac-w", type=int, default=8, help="Fixed-point frac width for GPT-2 scores")
    args = parser.parse_args()

    models = discover_models()
    if args.list_models:
        describe_models(models)
        return

    info = resolve_model(args.model, models)
    model_arg_path = Path(args.model) if args.model else None
    if (
        model_arg_path
        and model_arg_path.exists()
        and model_arg_path.is_file()
        and model_arg_path.suffix == ".gguf"
        and args.gguf is None
    ):
        gguf_path = model_arg_path
    else:
        gguf_path = select_gguf(info, args.gguf)
    weights_dir = select_weights(info, args.weights)

    if args.backend == "sw":
        if gguf_path is None and info.config_path:
            if args.prompt is None:
                raise RuntimeError("Adder SW inference requires --prompt")
            result = run_sw.run_sw_adder_inference(info.config_path, args.prompt)
            print(result)
            return
        if gguf_path is None:
            raise RuntimeError("SW inference requires a GGUF model")
        steps = args.steps if args.steps is not None else 16
        interactive = args.prompt is None
        run_sw.run_sw_inference(
            gguf_path,
            args.prompt or "",
            steps,
            args.do_sample,
            args.top_k,
            args.top_p,
            args.temperature,
            args.seed,
            interactive,
        )
        return

    if args.backend == "sv":
        if args.sv_impl == "adder" or (gguf_path is None and info.config_path):
            if weights_dir is None:
                raise RuntimeError("SV adder inference requires weights")
            if not info.config_path:
                raise RuntimeError("Missing config.json for adder model")
            result = run_hw.run_hw_adder_inference(
                weights_dir,
                info.config_path,
                args.prompt or "",
                args.steps,
            )
            print(result)
            return

        if gguf_path is None:
            raise RuntimeError("SV inference requires a GGUF model or --sv-impl adder")
        if weights_dir is None:
            raise RuntimeError("SV inference requires weights")

        steps = args.steps if args.steps is not None else 16
        interactive = args.prompt is None
        run_hw.run_hw_llm_inference(
            gguf_path,
            weights_dir,
            args.prompt or "",
            steps,
            args.do_sample,
            args.top_k,
            args.top_p,
            args.temperature,
            args.seed,
            args.frac_w,
            args.sv_impl,
            interactive,
        )
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
