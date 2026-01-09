#!/usr/bin/env bash
set -euo pipefail

verilator --sv --timing -Wall -Wno-fatal --top-module tb_top --binary \
  -I./src \
  ./tb/tb_top.sv

./obj_dir/Vtb_top "$@"
