#!/usr/bin/env bash
set -euo pipefail

verilator --sv --timing -Wall -Wno-fatal --top-module tb_infer --binary \
  -I./src \
  ./tb/tb_infer.sv

./obj_dir/Vtb_infer "$@"
