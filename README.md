# Transformer Accelerator (SystemVerilog)

A freakishly fun LLM accelerator in system verilog to run an entire LLM on your nice FPGA!!!



## Install

```
pip install gguf numpy
brew install verilator llama.cpp # https://verilator.org/guide/latest/install.html
```



## Overview
- Modular, fixed-point (Q8.8) transformer block skeleton.
- Includes Q/K/V projection, attention softmax, projection, MLP, and residuals.
- Host-mapped memory interface for inputs, weights, and outputs.
- Verilator testbench validates an identity behavior with zeroed weights.
- Matmul supports multi-lane MACs via `MATMUL_LANES` for DSP-heavy targets.

Memory Map (word addressed)
- 0x0000: input embeddings, size MAX_SEQ * D_MODEL
- 0x0100: Wq, size D_MODEL * D_MODEL
- 0x0200: Wk, size D_MODEL * D_MODEL
- 0x0300: Wv, size D_MODEL * D_MODEL
- 0x0400: Wo, size D_MODEL * D_MODEL
- 0x0500: W1, size D_MODEL * D_FF
- 0x0600: W2, size D_FF * D_MODEL
- 0x0700: output embeddings, size MAX_SEQ * D_MODEL (readback)
- 0x07F0: seq_len (write)
- 0x07F1: done (read)

### Run the Testbench
- ./tb/run_verilator.sh

### Load GGUF Weights (SmolLM2)
- Place the SmolLM2 model in `llm-models/`:
  - `llm-models/SmolLM2-135M-Instruct-Q8_0.gguf`
- Or download it:
  - `./llm-models/download-model.sh`
- Convert weights to mem files (requires `gguf` or `llama-cpp-python` Python package):
  - `python3 tools/gguf_export_full.py --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf --out llm-models/weights_full`
  - For FP16 mems: `python3 tools/gguf_export_full.py --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf --out llm-models/weights_full --fp16`
- Run the LLaMA behavioral SV model:
  - `./tb/run_llama_infer.sh +weights_dir=llm-models/weights_full +prompt_ids=tb/prompt_ids.mem +prompt_len=1`
  - For FP16 path: add `+use_fp16=1` and export FP16 mems first.

### Full Inference (Hello prompt)
- Export full weights to Q8.8 mem files:
  - `python3 tools/gguf_export_full.py --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf --out llm-models/weights_full`
- Run SV inference and compare to llama.cpp (plus a Python float reference):
  - `python3 tools/compare_hello.py --model-f16 llm-models/SmolLM2-135M-Instruct-Q8_0.gguf`
  - For FP16 mems: add `--use-fp16`

### LLaMA / SmolLM2 Export and SV Inference
- Export LLaMA weights (SmolLM2-135M example):
  - `python3 tools/gguf_export_full.py --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf --out llm-models/weights_full`
- Run the LLaMA behavioral SV model:
  - `./tb/run_llama_infer.sh +weights_dir=llm-models/weights_full +prompt_ids=tb/prompt_ids.mem +prompt_len=1`
- The LLaMA model uses RMSNorm + RoPE + GQA and expects the LLaMA-style `.mem` files.

### Chat With the HDL Model (Verilator)
- Single prompt:
  - `python3 tools/chat.py --prompt "1+1=?" --steps 30 --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf`
- Interactive:
  - `python3 tools/chat.py --steps 30 --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf`
- This uses Verilator inference; use `python3 tools/compare_hello.py` to compare HDL vs CPU outputs.
- Side-by-side per-token CPU comparison:
  - `python3 tools/chat.py --prompt "hello" --steps 8 --compare-cpu --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf`
  - For FP16 mems: add `--use-fp16`

### CPU-Only llama.cpp vs Verilog (Verilator)
- Run llama.cpp on CPU (one token):
  - `GGML_METAL_DISABLE=1 llama-cli --simple-io --no-display-prompt --single-turn --no-warmup --log-disable --device blas -m llm-models/SmolLM2-135M-Instruct-Q8_0.gguf -p "You are a helpful assistant. 1+1=?" -n 30 --temp 0 --top-k 1 --top-p 1 --repeat-penalty 1.0 --presence-penalty 0 --frequency-penalty 0 --seed 1 -ngl 0`
- Run Verilog (Verilator) on the same prompt:
  - `python3 tools/gguf_export_full.py --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf --out llm-models/weights_full`
  - `llama-tokenize --log-disable --ids -m llm-models/SmolLM2-135M-Instruct-Q8_0.gguf -p "hello"`
  - Write the IDs to `tb/prompt_ids.mem` (hex per line), then:
  - `./tb/run_llama_infer.sh +weights_dir=llm-models/weights_full +prompt_ids=tb/prompt_ids.mem +prompt_len=1`

### Measure Arithmetic Mismatch (SV vs Float)
- Reports SV top-5 logits vs Python float logits (absolute/relative error), plus llama.cpp next-token ID:
  - `python3 tools/measure_mismatch.py --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf --weights-dir llm-models/weights_full --prompt "hello"`
  - For FP16 mems: add `--use-fp16`

### Precision Debug (Layer Dumps)
- Dump one layer from SV (LLaMA path):
  - `./tb/run_llama_infer.sh +weights_dir=llm-models/weights_full +prompt_ids=tb/prompt_ids.mem +prompt_len=1 +dump_dir=tb/dump_sv +dump_layer=0 +dump_pos=0`
- Generate Python reference + compare against SV dumps:
  - `python3 tools/compare_layer.py --model llm-models/SmolLM2-135M-Instruct-Q8_0.gguf --prompt "hello" --layer 0 --pos 0 --sv-dir tb/dump_sv`
  - For FP16 dumps: export FP16 mems first, then add `+use_fp16=1` and pass `--sv-fp16 --fp16`

#### What the compare does
- Tokenizes “hello” with `llama-tokenize` to get prompt token IDs.
- Runs Verilator on the full SV model and prints `NEXT_TOKEN_ID`.
- Runs `llama-cli` greedy for 1 token and tokenizes that output.
- Runs a Python float reference with the same GGUF weights.
- Compares SV vs llama.cpp vs Python reference next-token IDs.

#### Manual inference run
- `./tb/run_infer.sh +weights_dir=llm-models/weights_full +prompt_ids=tb/prompt_ids.mem +prompt_len=<N>`
- Add `+dump_topk=1` to print the SV top-5 logits.

## Notes
- The math is simplified for clarity and simulation speed. Replace approximations with higher-accuracy units as needed.
- Adjust MAX_SEQ, D_MODEL, and D_FF in `src/transformer_accel.sv` for larger models.
- GGUF weights are down-projected to match the tiny accelerator dimensions (D_MODEL=4, D_FF=8).
- The converter expects the GGUF reader to return dequantized tensors; if your GGUF library returns raw quantized blocks, use a dequantizing reader.
- The full inference path uses a behavioral SV model and Q8.8 fixed-point math; expect differences from llama.cpp due to approximations.
- The TinyStories GGUF has 8 transformer blocks; the full inference path follows the model metadata.

### XCVU19P Preset
- Preset parameters and board info live in `src/board_presets.sv`.
- Use the preset top:
  - `src/transformer_accel_xcvu19p.sv`
- The preset increases `D_MODEL`, `D_FF`, and sets `MATMUL_LANES=512` to target DSP utilization.
- Update your weight export and memory map when changing these dimensions.
