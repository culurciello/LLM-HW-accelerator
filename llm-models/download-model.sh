#!/usr/bin/env bash
set -euo pipefail

# URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
URL="https://huggingface.co/afrideva/Tinystories-gpt-0.1-3m-GGUF/blob/main/tinystories-gpt-0.1-3m.fp16.gguf"
# OUT="SmolLM2-135M-Instruct-Q8_0.gguf"
OUT="tinystories-gpt-0.1-3m.fp16.gguf"


if command -v curl >/dev/null 2>&1; then
  curl -L -o "${OUT}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${OUT}" "${URL}"
else
  echo "error: need curl or wget to download the model" >&2
  exit 1
fi
