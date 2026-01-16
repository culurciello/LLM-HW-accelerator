# Numerical precision


In modern Transformer-based system like ChatGPT, the "brain" is not uniform. Some parts of the model are highly sensitive and require 16-bit (FP16/BF16) precision to avoid "breaking" the logic, while the heavy-lifting math modules are compressed to 8-bit (INT8/FP8) or lower to save speed.

Here is the breakdown of which modules run at which precision.

1. High-Precision Modules (16-bit)

These layers act as the "stabilizers" of the model. If you quantize these to 8-bit, the model often begins to hallucinate or lose its ability to follow instructions.

Layer Normalization (LayerNorm): These modules calculate the mean and variance of data. Because they deal with very small numbers and precise ratios, they almost always stay in 16-bit (or even 32-bit) to prevent "rounding to zero" errors.

Softmax (Attention Scores): In the self-attention mechanism, the model calculates how much "weight" to give each word. These scores are passed through a Softmax function. Small changes in these values lead to massive changes in the output, so this operation is kept in 16-bit.

Residual Connections: The "highway" paths that skip over layers to keep the signal strong are usually kept in 16-bit to ensure the gradient doesn't degrade.

Outlier Weights: In advanced systems like LLM.int8(), the model identifies "outlier" neurons that have unusually large values. These specific neurons (often less than 1% of the total) are kept in 16-bit while the rest are dropped to 8-bit.

2. Low-Precision Modules (8-bit / 4-bit)

These are the "muscle" of the model. They contain 95%+ of the parameters and do the heavy matrix multiplications.

Linear/Dense Layers (Feed-Forward Networks): The huge blocks of weights that process information between attention steps are the primary targets for 8-bit (or 4-bit) quantization.

Query, Key, and Value (QKV) Projections: The initial "sorting" of data into the attention mechanism is typically done in 8-bit.

Embedding Table: The massive dictionary that maps words to numbers is often stored in 8-bit to save several gigabytes of VRAM without affecting the model's logic.



### Table Summary

| Module / Layer Type | Common Precision | Notes                                         |
| ------------------- | ---------------- | --------------------------------------------- |
| Input Embeddings    | 8-bit / 4-bit    | Pure storage; low impact on reasoning.        |
| QKV Projections     | 8-bit / FP8      | High volume math; optimized for speed.        |
| Attention Softmax   | 16-bit           | Highly sensitive to small value changes.      |
| Feed-Forward (FFN)  | 8-bit / 4-bit    | The largest part of the model; must be small. |
| LayerNorm / RMSNorm | 16-bit           | Keeps the mathematical "range" stable.        |
| Residual Addition   | 16-bit           | Prevents "noise" from accumulating.           |
