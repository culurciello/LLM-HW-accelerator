#!/usr/bin/env bash
set -euo pipefail

verilator --sv --timing -Wall -Wno-fatal --top-module tb_gpt2_infer --binary \
  -I./src \
  ./tb/tb_gpt2_infer.sv

./obj_dir/Vtb_gpt2_infer "$@"
