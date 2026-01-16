# Transformer Accelerator (SystemVerilog)

A freakishly fun LLM accelerator in system verilog to run an entire LLM on your nice FPGA!!!

![](docs/cover.png)


## Project Status

| Test | Status |
| --- | --- |
| minGPT adder test | OK |
| SmolLM2-135M-Instruct-f16 test | TBD ?? |
| tinystories-gpt-0.1-3m.fp16 test | output YES but hard to compare to SW |


minGPT adder

```
$ python3 tools/compare_adder_sv_sw.py --prompt "3+4=
"
number of parameters: 0.09M
prompt: '3+4=\n'
ndigit: 2
inputs: 3 + 4
prompt_digits: [0, 3, 0, 4]
sv_out_digits_reversed: [7, 0, 0]
sw_out_digits_reversed: [7, 0, 0]
sv_out_digits_normal: [0, 0, 7]
sw_out_digits_normal: [0, 0, 7]
sv_sum: 7
sw_sum: 7
sv_vs_sw: MATCH
```


## Install

```
pip install gguf numpy
brew install verilator llama.cpp # https://verilator.org/guide/latest/install.html
```



## Overview
- Modular transformer block skeleton with an FP16 full-model path.
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

### Load GGUF Weights (SmolLM2 or TinyStories)
- Place the models in `llm-models/`:
  - `llm-models/SmolLM2-135M-Instruct-f16.gguf`
  - `llm-models/tinystories-gpt-0.1-3m.fp16.gguf`
- Or download SmolLM2:
  - `./llm-models/download-model.sh`
- Convert weights to mem files (requires `gguf` or `llama-cpp-python` Python package):
  - SmolLM2 (LLaMA path):
    - `python3 tools/gguf_export_full.py --model llm-models/SmolLM2-135M-Instruct-f16.gguf --out llm-models/weights_smol`
  - TinyStories GPT (GPT-2 path):
    - Fixed-point (legacy):
      - `python3 tools/gguf_export_full.py --model llm-models/tinystories-gpt-0.1-3m.fp16.gguf --out llm-models/weights_tinystories --fixed-point --frac-w 8`
    - FP16 (new SV path):
      - `python3 tools/gguf_export_full.py --model llm-models/tinystories-gpt-0.1-3m.fp16.gguf --out llm-models/weights_tinystories_fp16`
- Run the SV model:
  - LLaMA (SmolLM2):
    - `./tb/run_llama_infer.sh +weights_dir=llm-models/weights_smol +prompt_ids=tb/prompt_ids.mem +prompt_len=1`
  - GPT-2 (TinyStories):
    - `./tb/run_infer.sh +weights_dir=llm-models/weights_tinystories +prompt_ids=tb/prompt_ids.mem +prompt_len=1`

### Full Inference (Hello prompt)
- Export full weights to FP16 mem files:
  - `python3 tools/gguf_export_full.py --model llm-models/SmolLM2-135M-Instruct-f16.gguf --out llm-models/weights_smol`
- Run SV inference and compare to llama.cpp (plus a Python float reference):
  - `python3 tools/compare_hello.py --model-f16 llm-models/SmolLM2-135M-Instruct-f16.gguf --weights-dir llm-models/weights_smol`
  - GPT-2 fixed-point:
    - `python3 tools/compare_hello.py --model-f16 llm-models/tinystories-gpt-0.1-3m.fp16.gguf --weights-dir llm-models/weights_tinystories`
  - GPT-2 FP16:
    - `python3 tools/compare_hello.py --model-f16 llm-models/tinystories-gpt-0.1-3m.fp16.gguf --weights-dir llm-models/weights_tinystories_fp16 --gpt2-fp16`

### LLaMA / SmolLM2 Export and SV Inference
- Export LLaMA weights (SmolLM2-135M example):
  - `python3 tools/gguf_export_full.py --model llm-models/SmolLM2-135M-Instruct-f16.gguf --out llm-models/weights_smol`
- Run the LLaMA behavioral SV model:
  - `./tb/run_llama_infer.sh +weights_dir=llm-models/weights_smol +prompt_ids=tb/prompt_ids.mem +prompt_len=1`
- The LLaMA model uses RMSNorm + RoPE + GQA and expects the LLaMA-style `.mem` files.

### Adder (minGPT) Export and SV Inference
Train a model from https://github.com/karpathy/minGPT: ```$ python3 projects/adder/adder.py ```.
Then:
- Export the minGPT checkpoint to SV mem files:
  - `python3 tools/mingpt_export_sv.py --checkpoint llm-models/adder/model.pt --config llm-models/adder/config.json --out llm-models/adder/weights_sv`
- Create a prompt (digits only, one hex ID per line). Example for 85 + 50 (prompt digits `8 5 5 0`):
  - `printf "00000008\n00000005\n00000005\n00000000\n" > tb/prompt_ids.mem`
- Run the SV model (one next token per run):
  - `./tb/run_adder_infer.sh +weights_dir=llm-models/adder/weights_sv +prompt_ids=tb/prompt_ids.mem +prompt_len=4`
- Repeat with the new token appended to build the full (n+1)-digit sum.

### Chat With the HDL Model (Verilator)
- Single prompt:
  - `python3 tools/chat.py --prompt "1+1=?" --steps 30 --model llm-models/SmolLM2-135M-Instruct-f16.gguf`
- Interactive:
  - `python3 tools/chat.py --steps 30 --model llm-models/SmolLM2-135M-Instruct-f16.gguf`
- Sampling (top-k/top-p/temperature) for TinyStories:
  - `python3 tools/chat.py --model llm-models/tinystories-gpt-0.1-3m.fp16.gguf --weights-dir llm-models/weights_tinystories --prompt "<|start_story|>Once upon a time, " --steps 200 --do-sample --top-k 40 --top-p 0.9 --temperature 0.6`
- Sampling (top-k/top-p/temperature) for TinyStories FP16 SV:
  - `python3 tools/chat.py --model llm-models/tinystories-gpt-0.1-3m.fp16.gguf --weights-dir llm-models/weights_tinystories_fp16 --prompt "<|start_story|>Once upon a time, " --steps 200 --do-sample --top-k 40 --top-p 0.9 --temperature 0.6 --gpt2-fp16`
- Sampling (top-k/top-p/temperature) with CPU reference (same settings):
  - `python3 tools/tinystories_generate.py --model llm-models/tinystories-gpt-0.1-3m.fp16.gguf --prompt "<|start_story|>Once upon a time, " --n-predict 200`
- This uses Verilator inference; use `python3 tools/compare_hello.py` to compare HDL vs CPU outputs.
- Side-by-side per-token CPU comparison:
  - `python3 tools/chat.py --prompt "hello" --steps 8 --compare-cpu --model llm-models/SmolLM2-135M-Instruct-f16.gguf`

### CPU-Only llama.cpp vs Verilog (Verilator)
- Run llama.cpp on CPU (one token):
  - `GGML_METAL_DISABLE=1 llama-cli --simple-io --no-display-prompt --single-turn --no-warmup --log-disable --device blas -m llm-models/SmolLM2-135M-Instruct-f16.gguf -p "1+1=?" -n 30 --temp 0 --top-k 1 --top-p 1 --repeat-penalty 1.0 --presence-penalty 0 --frequency-penalty 0 --seed 1 -ngl 0`
- Run Verilog (Verilator) on the same prompt:
  - `python3 tools/gguf_export_full.py --model llm-models/SmolLM2-135M-Instruct-f16.gguf --out llm-models/weights_smol`
  - `llama-tokenize --log-disable --ids -m llm-models/SmolLM2-135M-Instruct-f16.gguf -p "hello"`
  - Write the IDs to `tb/prompt_ids.mem` (hex per line), then:
  - `./tb/run_llama_infer.sh +weights_dir=llm-models/weights_smol +prompt_ids=tb/prompt_ids.mem +prompt_len=1`

### Measure Arithmetic Mismatch (SV vs Float)
- Reports SV top-5 logits vs Python float logits (absolute/relative error), plus llama.cpp next-token ID:
  - `python3 tools/measure_mismatch.py --model llm-models/SmolLM2-135M-Instruct-f16.gguf --weights-dir llm-models/weights_smol --prompt "hello"`

### Precision Debug (Layer Dumps)
- Dump one layer from SV (LLaMA path):
  - `./tb/run_llama_infer.sh +weights_dir=llm-models/weights_smol +prompt_ids=tb/prompt_ids.mem +prompt_len=1 +dump_dir=tb/dump_sv +dump_layer=0 +dump_pos=0`
- Generate Python reference + compare against SV dumps:
  - `python3 tools/compare_layer.py --model llm-models/SmolLM2-135M-Instruct-f16.gguf --prompt "hello" --layer 0 --pos 0 --sv-dir tb/dump_sv`
- Or generate fresh SV dumps automatically:
  - `python3 tools/compare_layer.py --model llm-models/SmolLM2-135M-Instruct-f16.gguf --prompt "hello" --layer 0 --pos 0 --sv-dir tb/dump_sv --run-sv`

### TinyStories Generation Test
- Run the TinyStories GPT model with the recommended sampling config and template:
  - `python3 tools/tinystories_generate.py --model llm-models/tinystories-gpt-0.1-3m.fp16.gguf`
- SW reference generation (generic):
  - `python3 tools/generate_sw.py --model llm-models/tinystories-gpt-0.1-3m.fp16.gguf --prompt "<|start_story|>Once upon a time, " --n-predict 200`
- SV/HW generation (generic):
  - `python3 tools/generate_sv.py --model llm-models/tinystories-gpt-0.1-3m.fp16.gguf --weights-dir llm-models/weights_tinystories_fp16 --prompt "<|start_story|>Once upon a time, " --steps 200 --gpt2-fp16`

#### What the compare does
- Tokenizes “hello” with `llama-tokenize` to get prompt token IDs.
- Runs Verilator on the full SV model and prints `NEXT_TOKEN_ID`.
- Runs `llama-cli` greedy for 1 token and tokenizes that output.
- Runs a Python float reference with the same GGUF weights.
- Compares SV vs llama.cpp vs Python reference next-token IDs.

#### Manual inference run
- `./tb/run_llama_infer.sh +weights_dir=llm-models/weights_smol +prompt_ids=tb/prompt_ids.mem +prompt_len=<N>`
- Add `+dump_topk=1` to print the SV top-40 logits.
 - GPT-2 FP16:
   - `./tb/run_gpt2_infer.sh +weights_dir=llm-models/weights_tinystories_fp16 +prompt_ids=tb/prompt_ids.mem +prompt_len=<N> +dump_topk=1`

## Notes
- The math is simplified for clarity and simulation speed. Replace approximations with higher-accuracy units as needed.
- Adjust MAX_SEQ, D_MODEL, and D_FF in `src/transformer_accel.sv` for larger models.
- GGUF weights are down-projected to match the tiny accelerator dimensions (D_MODEL=4, D_FF=8).
- Quantized GGUF weights are not supported; use the F16 model.
- The full inference path uses a behavioral SV model and FP16 math; expect differences from llama.cpp due to approximations.
- The TinyStories GGUF has 8 transformer blocks; the full inference path follows the model metadata.

### XCVU19P Preset
- Preset parameters and board info live in `src/board_presets.sv`.
- Use the preset top:
  - `src/transformer_accel_xcvu19p.sv`
- The preset increases `D_MODEL`, `D_FF`, and sets `MATMUL_LANES=512` to target DSP utilization.
- Update your weight export and memory map when changing these dimensions.



## Models

- https://www.reddit.com/r/LocalLLaMA/comments/1gnws1h/smallest_llamacpp_model/

- https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF

- https://huggingface.co/afrideva/Tinystories-gpt-0.1-3m-GGUF/tree/main
