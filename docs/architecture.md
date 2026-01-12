# System Architecture Overview

This repository contains two complementary implementations:

- A modular, hardware-structured accelerator (`src/transformer_accel.sv`) with reusable compute blocks.
- A behavioral full-model reference (`src/transformer_full.sv`) used for GPT-2 style end-to-end testing.
- A behavioral LLaMA/GQA model (`src/llama_full.sv`) used for SmolLM2-style testing.

Both implement a GPT2-style transformer, but the accelerator version is built around explicit matmul, softmax, and layernorm modules so it maps cleanly to FPGA resources. The full-model path now runs in FP16 for closer alignment with the reference model.

## Top-Level Blocks

### `transformer_accel.sv`
The accelerator is a single transformer block pipeline:

1. **Input embedding buffer** (host-writable memory)
2. **Q/K/V projections** via `matmul` modules
3. **Attention scores** with a Q * K^T matmul
4. **Softmax** over attention scores (`softmax.sv`)
5. **Context matmul** (probabilities * V)
6. **Output projection** (Wo)
7. **Residual + MLP** (W1, activation, W2)
8. **Final output buffer** (host-readable memory)

All intermediate tensors are stored in on-chip arrays (modeled as SV memories for synthesis).

### `matmul.sv`
The matmul block is the primary compute engine:

- Parameterized by M, K, N at runtime.
- FP16 MAC accumulation (`DATA_W`, `ACC_W`).
- `LANES` parameter enables multi-lane MACs to scale DSP usage.
- `b_transpose` allows K^T access without an explicit transpose buffer.

The `MATMUL_LANES` parameter in `transformer_accel.sv` controls how many parallel MAC lanes are used. On large devices (e.g., XCVU19P), this is set high to match available DSP slices.

### `softmax.sv`
Implements a fixed-point softmax across attention score rows. It uses a stabilized exp approximation and normalizes each row. This is simplified for clarity and simulation speed.

### `layernorm.sv`
Implements fixed-point layer normalization for the accelerator block. The full-model path uses similar math but is implemented directly in `transformer_full.sv`.

### `transformer_full.sv`
Behavioral, layer-by-layer full-model implementation used for end-to-end inference:

- Implements all layers, embeddings, KV cache, and output projection.
- Uses GGUF-exported weights from `tools/gguf_export_full.py`.
- Runs entirely inside a single `always_ff` for Verilator simulation.

This is not a high-performance microarchitecture, but it is a functional reference for testing.

### `llama_full.sv`
Behavioral LLaMA/GQA implementation for SmolLM2-style models:

- RMSNorm instead of LayerNorm.
- RoPE positional encoding applied to Q and K.
- Separate Q/K/V weights with grouped query attention (N_HEAD != N_KV_HEAD).
- Gated MLP (SiLU(gate) * up) + down projection.

## Dataflow and Memory

### Host-Visible Memory Map (accelerator)
The accelerator exposes a simple word-addressed map for inputs, weights, and outputs:

- `0x0000`: input embeddings (`MAX_SEQ * D_MODEL`)
- `0x0100`: Wq (`D_MODEL * D_MODEL`)
- `0x0200`: Wk (`D_MODEL * D_MODEL`)
- `0x0300`: Wv (`D_MODEL * D_MODEL`)
- `0x0400`: Wo (`D_MODEL * D_MODEL`)
- `0x0500`: W1 (`D_MODEL * D_FF`)
- `0x0600`: W2 (`D_FF * D_MODEL`)
- `0x0700`: output embeddings
- `0x07F0`: control (`seq_len`, `done`)

### Weight Export
`tools/gguf_export_full.py` exports GGUF weights into FP16 `.mem` files. These are loaded by Verilator testbenches and can be adapted for hardware integration.

## FP16 Arithmetic

- **Format**: FP16 for weights and activations.
- **Accumulators**: wider (`ACC_W`) to reduce overflow.
- **Approximations**: exp/softmax approximations are used to keep logic simple.

Expect differences versus llama.cpp due to approximations and simplified control flow.

## DSP Scaling

The accelerator’s matmul units scale with DSP availability by increasing `MATMUL_LANES`:

- `MATMUL_LANES = 1` is a single MAC per cycle.
- Larger values perform multiple MACs per cycle, using more DSP slices.

The XCVU19P preset in `src/board_presets.sv` sets `MATMUL_LANES=512` and larger tensor sizes to match the DSP-rich target.

## Verification and Test Strategy

- `tb/tb_top.sv` exercises the modular accelerator with small dimensions.
- `tb/tb_infer.sv` drives the full-model SV and produces next-token IDs.
- `tools/compare_hello.py` compares Verilator output to a float reference and llama.cpp.
- `tools/measure_mismatch.py` quantifies per-token mismatch in top‑k logits.

## Presets

- `src/board_presets.sv` contains board-scale parameters.
- `src/transformer_accel_xcvu19p.sv` instantiates a preset for XCVU19P.

When you change dimensions (D_MODEL, D_FF, MAX_SEQ), you must regenerate weights and update memory sizes accordingly.
