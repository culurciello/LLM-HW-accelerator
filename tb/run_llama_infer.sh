#!/usr/bin/env bash
set -euo pipefail

verilator --sv --timing -Wall -Wno-fatal --top-module tb_llama_infer --binary \
  -I./src \
  ./tb/tb_llama_infer.sv

./obj_dir/Vtb_llama_infer "$@"
