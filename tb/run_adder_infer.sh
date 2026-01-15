#!/usr/bin/env bash
set -euo pipefail

verilator --sv --timing -Wall -Wno-fatal --top-module tb_adder_infer --binary \
  -I./src \
  ./tb/tb_adder_infer.sv

./obj_dir/Vtb_adder_infer "$@"
