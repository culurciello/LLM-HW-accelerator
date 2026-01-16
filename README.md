# Transformer Accelerator (SystemVerilog)

A freakishly fun LLM accelerator in system verilog to run an entire LLM on your nice FPGA!!!

![](docs/cover.png)


## Project Status

| Test | Status | To Do |
| --- | --- | --- |
| minGPT adder test | OK | |
| tinystories-gpt-0.1-3m.fp16 test | output YES | test accuracy layer-by-layer| 
| SmolLM2-135M-Instruct-f16 test |  output YES | test accuracy layer-by-layer| 


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

## Models and Usage

Models live under `llm-models/<name>/`. Use the unified runner to list or select them:

```
python3 tools/chat.py --list-models
```

### Adder (minGPT)

Train a model from https://github.com/karpathy/minGPT: $ python3 projects/adder/adder.py . Then:

- Export the minGPT checkpoint to SV mem files:
`python3 tools/mingpt_export_sv.py --checkpoint llm-models/adder/model.pt --config llm-models/adder/config.json --out llm-models/adder/weights_sv`

- Layout: `llm-models/adder/` with `config.json`, `model.pt`, and `weights_sv/`.


- RUN Hardware (system-verilog or SV):
`python3 tools/chat.py --model adder --backend sv --sv-impl adder --prompt "3+4="`

- RUN Software (SW llama.cpp):
`python3 tools/chat.py --model adder --backend sw --prompt "3+4="`

### TinyStories (GPT-2)
- Layout: `llm-models/tinystories/` with `tinystories-gpt-0.1-3m.fp16.gguf` and `weights_tinystories_fp16/`.

- RUN Hardware (system-verilog or SV):
```
python3 tools/chat.py --model tinystories --backend sv --prompt "<|start_story|>Once upon a time, " --steps 20 --do-sample --top-k 40 --top-p 0.9 --temperature 0.6
```

- RUN Software (SW llama.cpp):
```
python3 tools/chat.py --model tinystories --backend sw --prompt "<|start_story|>Once upon a time, " --steps 20 --do-sample --top-k 40 --top-p 0.9 --temperature 0.6
```

### SmolLM2 (LLaMA)
- Layout: `llm-models/smollm2/` with `SmolLM2-135M-Instruct-f16.gguf` and `weights_smol/`.

- RUN Hardware (system-verilog or SV):
```
python3 tools/chat.py --model smollm2 --backend sv --prompt "hello" --steps 30
```

- RUN Software (SW llama.cpp):
```
python3 tools/chat.py --model smollm2 --backend sw --prompt "hello" --steps 30
```

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
