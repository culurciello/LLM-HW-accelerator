`timescale 1ns/1ps
`include "transformer_pkg.sv"

module gpt2_full_fp16 #(
    parameter int DATA_W = transformer_pkg::DATA_W,
    parameter int ACC_W  = transformer_pkg::ACC_W,
    parameter int N_LAYER = 8,
    parameter int N_HEAD  = 16,
    parameter int D_MODEL = 64,
    parameter int D_FF    = 256,
    parameter int MAX_CTX = 2048,
    parameter int VOCAB   = 50257,
    parameter int MAX_PROMPT = 512,
    parameter real LN_EPS = 1e-5,
    parameter int TOP_K = 40
  )(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  int unsigned prompt_len,
    output logic done,
    output logic [31:0] next_token_id
  );

  import transformer_pkg::*;

  localparam int HEAD_DIM = D_MODEL / N_HEAD;

  logic [DATA_W-1:0] token_embd [0:VOCAB*D_MODEL-1];
  logic [DATA_W-1:0] pos_embd   [0:MAX_CTX*D_MODEL-1];
  logic [DATA_W-1:0] out_weight [0:VOCAB*D_MODEL-1];
  logic [DATA_W-1:0] out_norm_w [0:D_MODEL-1];
  logic [DATA_W-1:0] out_norm_b [0:D_MODEL-1];

  logic [DATA_W-1:0] attn_norm_w [0:N_LAYER*D_MODEL-1];
  logic [DATA_W-1:0] attn_norm_b [0:N_LAYER*D_MODEL-1];
  logic [DATA_W-1:0] qkv_w       [0:N_LAYER*D_MODEL*3*D_MODEL-1];
  logic [DATA_W-1:0] qkv_b       [0:N_LAYER*3*D_MODEL-1];
  logic [DATA_W-1:0] attn_out_w  [0:N_LAYER*D_MODEL*D_MODEL-1];
  logic [DATA_W-1:0] attn_out_b  [0:N_LAYER*D_MODEL-1];
  logic [DATA_W-1:0] ffn_norm_w  [0:N_LAYER*D_MODEL-1];
  logic [DATA_W-1:0] ffn_norm_b  [0:N_LAYER*D_MODEL-1];
  logic [DATA_W-1:0] ffn_up_w    [0:N_LAYER*D_MODEL*D_FF-1];
  logic [DATA_W-1:0] ffn_up_b    [0:N_LAYER*D_FF-1];
  logic [DATA_W-1:0] ffn_dn_w    [0:N_LAYER*D_FF*D_MODEL-1];
  logic [DATA_W-1:0] ffn_dn_b    [0:N_LAYER*D_MODEL-1];

  logic [31:0] input_tokens [0:MAX_PROMPT-1];

  logic [DATA_W-1:0] k_cache [0:N_LAYER-1][0:MAX_CTX*D_MODEL-1];
  logic [DATA_W-1:0] v_cache [0:N_LAYER-1][0:MAX_CTX*D_MODEL-1];

  logic [DATA_W-1:0] x        [0:D_MODEL-1];
  logic [DATA_W-1:0] x_norm   [0:D_MODEL-1];
  logic [DATA_W-1:0] qkv      [0:3*D_MODEL-1];
  logic [DATA_W-1:0] attn_out [0:D_MODEL-1];
  logic [DATA_W-1:0] attn_proj[0:D_MODEL-1];
  logic [DATA_W-1:0] ffn_up   [0:D_FF-1];
  logic [DATA_W-1:0] ffn_act  [0:D_FF-1];
  logic [DATA_W-1:0] ffn_out  [0:D_MODEL-1];

  logic [31:0] debug_topk_id [0:TOP_K-1];
  logic signed [ACC_W-1:0] debug_topk_score [0:TOP_K-1];

  function automatic int unsigned idx2d(
      input int unsigned row,
      input int unsigned col,
      input int unsigned stride
  );
    return row * stride + col;
  endfunction

  function automatic logic [15:0] real_to_fp16(input real val_r);
    integer sign;
    integer exp;
    real mant;
    integer man_bits;
    begin
      if (val_r == 0.0) begin
        real_to_fp16 = 16'h0000;
      end else begin
        sign = (val_r < 0.0);
        if (val_r < 0.0)
          val_r = -val_r;
        exp = $rtoi($floor($ln(val_r) / $ln(2.0)));
        mant = val_r / (2.0 ** exp);
        man_bits = $rtoi((mant - 1.0) * 1024.0 + 0.5);
        if (man_bits < 0)
          man_bits = 0;
        if (man_bits > 1023)
          man_bits = 1023;
        exp = exp + 15;
        if (exp <= 0) begin
          real_to_fp16 = 16'h0000;
        end else if (exp >= 31) begin
          real_to_fp16 = {sign[0], 5'h1F, 10'h000};
        end else begin
          real_to_fp16 = {sign[0], exp[4:0], man_bits[9:0]};
        end
      end
    end
  endfunction

  function automatic real fp16_to_real(input logic [15:0] h);
    integer sign;
    integer exp;
    integer man;
    real mant;
    begin
      if (h[14:0] == 0) begin
        fp16_to_real = 0.0;
      end else begin
        sign = h[15];
        exp = h[14:10] - 15;
        man = h[9:0];
        mant = 1.0 + (man / 1024.0);
        fp16_to_real = mant * (2.0 ** exp);
        if (sign)
          fp16_to_real = -fp16_to_real;
      end
    end
  endfunction

  function automatic logic [15:0] gelu_fp16(
      input logic [15:0] x_in
  );
    real x;
    real inner;
    begin
      x = fp16_to_real(x_in);
      inner = 0.5 * x * (1.0 + $tanh($sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * x * x * x)));
      gelu_fp16 = real_to_fp16(inner);
    end
  endfunction

  task automatic layer_norm_fp16(
      input  int layer,
      input  logic [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out_vec[0:D_MODEL-1],
      input  logic [DATA_W-1:0] weight [0:N_LAYER*D_MODEL-1],
      input  logic [DATA_W-1:0] bias   [0:N_LAYER*D_MODEL-1]
  );
    integer i;
    real mean;
    real variance;
    real inv_std;
    real val;
    begin
      mean = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1)
        mean = mean + fp16_to_real(in_vec[i]);
      mean = mean / D_MODEL;
      variance = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = fp16_to_real(in_vec[i]) - mean;
        variance = variance + val * val;
      end
      variance = variance / D_MODEL;
      inv_std = 1.0 / $sqrt(variance + LN_EPS);
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = (fp16_to_real(in_vec[i]) - mean) * inv_std;
        val = val * fp16_to_real(weight[layer * D_MODEL + i]) +
              fp16_to_real(bias[layer * D_MODEL + i]);
        out_vec[i] = real_to_fp16(val);
      end
    end
  endtask

  task automatic layer_norm_out_fp16(
      input  logic [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out_vec[0:D_MODEL-1]
  );
    integer i;
    real mean;
    real variance;
    real inv_std;
    real val;
    begin
      mean = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1)
        mean = mean + fp16_to_real(in_vec[i]);
      mean = mean / D_MODEL;
      variance = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = fp16_to_real(in_vec[i]) - mean;
        variance = variance + val * val;
      end
      variance = variance / D_MODEL;
      inv_std = 1.0 / $sqrt(variance + LN_EPS);
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = (fp16_to_real(in_vec[i]) - mean) * inv_std;
        val = val * fp16_to_real(out_norm_w[i]) + fp16_to_real(out_norm_b[i]);
        out_vec[i] = real_to_fp16(val);
      end
    end
  endtask

  task automatic matvec_qkv_fp16(
      input  int layer,
      input  logic [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out [0:3*D_MODEL-1]
  );
    integer i;
    integer j;
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < 3*D_MODEL; j = j + 1) begin
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          prod_fp16 = fp16_mul(vec[i], qkv_w[layer*D_MODEL*3*D_MODEL + i*3*D_MODEL + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = fp16_add(acc_fp16, qkv_b[layer*3*D_MODEL + j]);
      end
    end
  endtask

  task automatic matvec_attn_out_fp16(
      input  int layer,
      input  logic [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out [0:D_MODEL-1]
  );
    integer i;
    integer j;
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_MODEL; j = j + 1) begin
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          prod_fp16 = fp16_mul(vec[i], attn_out_w[layer*D_MODEL*D_MODEL + i*D_MODEL + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = fp16_add(acc_fp16, attn_out_b[layer*D_MODEL + j]);
      end
    end
  endtask

  task automatic matvec_ffn_up_fp16(
      input  int layer,
      input  logic [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out [0:D_FF-1]
  );
    integer i;
    integer j;
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_FF; j = j + 1) begin
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          prod_fp16 = fp16_mul(vec[i], ffn_up_w[layer*D_MODEL*D_FF + i*D_FF + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = fp16_add(acc_fp16, ffn_up_b[layer*D_FF + j]);
      end
    end
  endtask

  task automatic matvec_ffn_down_fp16(
      input  int layer,
      input  logic [DATA_W-1:0] vec [0:D_FF-1],
      output logic [DATA_W-1:0] out [0:D_MODEL-1]
  );
    integer i;
    integer j;
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_MODEL; j = j + 1) begin
        acc_fp16 = '0;
        for (i = 0; i < D_FF; i = i + 1) begin
          prod_fp16 = fp16_mul(vec[i], ffn_dn_w[layer*D_FF*D_MODEL + i*D_MODEL + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = fp16_add(acc_fp16, ffn_dn_b[layer*D_MODEL + j]);
      end
    end
  endtask

  task automatic attention_fp16(
      input int layer,
      input int pos,
      input logic [DATA_W-1:0] qkv_vec [0:3*D_MODEL-1],
      output logic [DATA_W-1:0] out [0:D_MODEL-1]
  );
    integer h;
    integer t;
    integer d;
    integer i;
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    real scores_r [0:MAX_CTX-1];
    real weight_r [0:MAX_CTX-1];
    real max_score;
    real sum_exp;
    real scale;
    begin
      scale = 1.0 / $sqrt(HEAD_DIM);
      for (i = 0; i < D_MODEL; i = i + 1)
        out[i] = '0;

      for (h = 0; h < N_HEAD; h = h + 1) begin
        for (t = 0; t <= pos; t = t + 1) begin
          acc_fp16 = '0;
          for (d = 0; d < HEAD_DIM; d = d + 1) begin
            i = h * HEAD_DIM + d;
            prod_fp16 = fp16_mul(qkv_vec[i], k_cache[layer][idx2d(t, i, D_MODEL)]);
            acc_fp16 = fp16_add(acc_fp16, prod_fp16);
          end
          scores_r[t] = fp16_to_real(acc_fp16) * scale;
        end
        max_score = -1.0e30;
        for (t = 0; t <= pos; t = t + 1) begin
          if (scores_r[t] > max_score)
            max_score = scores_r[t];
        end
        sum_exp = 0.0;
        for (t = 0; t <= pos; t = t + 1) begin
          weight_r[t] = $exp(scores_r[t] - max_score);
          sum_exp = sum_exp + weight_r[t];
        end
        for (d = 0; d < HEAD_DIM; d = d + 1) begin
          acc_fp16 = '0;
          for (t = 0; t <= pos; t = t + 1) begin
            i = h * HEAD_DIM + d;
            prod_fp16 = fp16_mul(real_to_fp16(weight_r[t] / sum_exp),
                                 v_cache[layer][idx2d(t, i, D_MODEL)]);
            acc_fp16 = fp16_add(acc_fp16, prod_fp16);
          end
          out[h * HEAD_DIM + d] = acc_fp16;
        end
      end
    end
  endtask

  task automatic compute_logits_fp16(
      input  logic [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic [31:0] token_id
  );
    integer tok;
    integer d;
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    real acc_r;
    real top_r [0:TOP_K-1];
    logic [31:0] top_id [0:TOP_K-1];
    logic [15:0] score_fp16;
    integer k_idx;
    integer insert;
    begin
      token_id = 0;
      for (k_idx = 0; k_idx < TOP_K; k_idx = k_idx + 1) begin
        top_r[k_idx] = -1.0e30;
        top_id[k_idx] = 0;
      end
      for (tok = 0; tok < VOCAB; tok = tok + 1) begin
        acc_fp16 = '0;
        for (d = 0; d < D_MODEL; d = d + 1) begin
          prod_fp16 = fp16_mul(vec[d], out_weight[idx2d(tok, d, D_MODEL)]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        acc_r = fp16_to_real(acc_fp16);
        if (acc_r > top_r[TOP_K-1]) begin
          insert = TOP_K-1;
          while (insert > 0 && acc_r > top_r[insert-1]) begin
            top_r[insert] = top_r[insert-1];
            top_id[insert] = top_id[insert-1];
            insert = insert - 1;
          end
          top_r[insert] = acc_r;
          top_id[insert] = tok[31:0];
        end
      end
      token_id = top_id[0];
      for (k_idx = 0; k_idx < TOP_K; k_idx = k_idx + 1) begin
        debug_topk_id[k_idx] = top_id[k_idx];
        score_fp16 = real_to_fp16(top_r[k_idx]);
        debug_topk_score[k_idx] = {{(ACC_W-16){score_fp16[15]}}, score_fp16};
      end
    end
  endtask

  integer pos;
  integer layer;
  integer i;

  /* verilator lint_off BLKSEQ */
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done <= 1'b0;
      next_token_id <= 0;
    end else begin
      done <= 1'b0;
      if (start) begin
        for (layer = 0; layer < N_LAYER; layer = layer + 1) begin
          for (pos = 0; pos < MAX_CTX; pos = pos + 1) begin
            for (i = 0; i < D_MODEL; i = i + 1) begin
              k_cache[layer][idx2d(pos, i, D_MODEL)] <= '0;
              v_cache[layer][idx2d(pos, i, D_MODEL)] <= '0;
            end
          end
        end

        for (pos = 0; pos < prompt_len; pos = pos + 1) begin
          for (i = 0; i < D_MODEL; i = i + 1) begin
            x[i] = fp16_add(
              token_embd[idx2d(input_tokens[pos], i, D_MODEL)],
              pos_embd[idx2d(pos, i, D_MODEL)]
            );
          end
          for (layer = 0; layer < N_LAYER; layer = layer + 1) begin
            layer_norm_fp16(layer, x, x_norm, attn_norm_w, attn_norm_b);
            matvec_qkv_fp16(layer, x_norm, qkv);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              k_cache[layer][idx2d(pos, i, D_MODEL)] = qkv[D_MODEL + i];
              v_cache[layer][idx2d(pos, i, D_MODEL)] = qkv[2*D_MODEL + i];
            end
            attention_fp16(layer, pos, qkv, attn_out);
            matvec_attn_out_fp16(layer, attn_out, attn_proj);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              x[i] = fp16_add(x[i], attn_proj[i]);
            end
            layer_norm_fp16(layer, x, x_norm, ffn_norm_w, ffn_norm_b);
            matvec_ffn_up_fp16(layer, x_norm, ffn_up);
            for (i = 0; i < D_FF; i = i + 1) begin
              ffn_act[i] = gelu_fp16(ffn_up[i]);
            end
            matvec_ffn_down_fp16(layer, ffn_act, ffn_out);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              x[i] = fp16_add(x[i], ffn_out[i]);
            end
          end
        end

        layer_norm_out_fp16(x, x_norm);
        compute_logits_fp16(x_norm, next_token_id);
        done <= 1'b1;
      end
    end
  end
  /* verilator lint_on BLKSEQ */
endmodule
