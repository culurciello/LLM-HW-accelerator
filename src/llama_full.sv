`timescale 1ns/1ps
`include "transformer_pkg.sv"

module llama_full #(
    parameter int DATA_W = transformer_pkg::DATA_W,
    parameter int FRAC_W = transformer_pkg::FRAC_W,
    parameter int ACC_W  = transformer_pkg::ACC_W,
    parameter int N_LAYER = 30,
    parameter int N_HEAD  = 9,
    parameter int N_KV_HEAD = 3,
    parameter int D_MODEL = 576,
    parameter int D_FF    = 1536,
    parameter int MAX_CTX = 8192,
    parameter int VOCAB   = 49152,
    parameter int ROPE_DIM = 64,
    parameter real ROPE_BASE = 100000.0,
    parameter real RMS_EPS = 1e-5
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
  localparam int D_KV = N_KV_HEAD * HEAD_DIM;

  logic signed [DATA_W-1:0] token_embd [0:VOCAB*D_MODEL-1];
  logic signed [DATA_W-1:0] out_weight [0:VOCAB*D_MODEL-1];
  logic signed [DATA_W-1:0] out_norm_w [0:D_MODEL-1];

  logic signed [DATA_W-1:0] attn_norm_w [0:N_LAYER*D_MODEL-1];
  logic signed [DATA_W-1:0] ffn_norm_w  [0:N_LAYER*D_MODEL-1];

  logic signed [DATA_W-1:0] attn_q_w    [0:N_LAYER*D_MODEL*D_MODEL-1];
  logic signed [DATA_W-1:0] attn_k_w    [0:N_LAYER*D_MODEL*D_KV-1];
  logic signed [DATA_W-1:0] attn_v_w    [0:N_LAYER*D_MODEL*D_KV-1];
  logic signed [DATA_W-1:0] attn_out_w  [0:N_LAYER*D_MODEL*D_MODEL-1];

  logic signed [DATA_W-1:0] ffn_gate_w  [0:N_LAYER*D_MODEL*D_FF-1];
  logic signed [DATA_W-1:0] ffn_up_w    [0:N_LAYER*D_MODEL*D_FF-1];
  logic signed [DATA_W-1:0] ffn_dn_w    [0:N_LAYER*D_FF*D_MODEL-1];

  logic signed [DATA_W-1:0] k_cache [0:N_LAYER-1][0:MAX_CTX*D_KV-1];
  logic signed [DATA_W-1:0] v_cache [0:N_LAYER-1][0:MAX_CTX*D_KV-1];

  logic [31:0] input_tokens [0:MAX_CTX-1];
  logic signed [DATA_W-1:0] x        [0:D_MODEL-1];
  logic signed [DATA_W-1:0] x_norm   [0:D_MODEL-1];
  logic signed [DATA_W-1:0] q        [0:D_MODEL-1];
  logic signed [DATA_W-1:0] k        [0:D_KV-1];
  logic signed [DATA_W-1:0] v        [0:D_KV-1];
  logic signed [DATA_W-1:0] attn_out [0:D_MODEL-1];
  logic signed [DATA_W-1:0] attn_proj[0:D_MODEL-1];
  logic signed [DATA_W-1:0] ffn_gate [0:D_FF-1];
  logic signed [DATA_W-1:0] ffn_up   [0:D_FF-1];
  logic signed [DATA_W-1:0] ffn_act  [0:D_FF-1];
  logic signed [DATA_W-1:0] ffn_out  [0:D_MODEL-1];

  logic [31:0] debug_topk_id [0:4];
  logic signed [ACC_W-1:0] debug_topk_score [0:4];

  function automatic int idx2d(input int row, input int col, input int stride);
    begin
      idx2d = row * stride + col;
    end
  endfunction

  function automatic real q_to_real(input logic signed [DATA_W-1:0] v);
    begin
      q_to_real = v / real'(1 << FRAC_W);
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] real_to_q(input real v);
    integer tmp;
    begin
      tmp = $rtoi(v * (1 << FRAC_W));
      real_to_q = sat_q(tmp);
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] silu_approx(
      input logic signed [DATA_W-1:0] x_in
  );
    logic signed [DATA_W-1:0] one_q;
    logic signed [DATA_W-1:0] exp_neg;
    logic signed [DATA_W-1:0] denom;
    logic signed [DATA_W-1:0] frac;
    begin
      one_q = (1 <<< FRAC_W);
      exp_neg = exp_approx(-x_in);
      denom = add_q(one_q, exp_neg);
      frac = div_q(x_in, denom);
      silu_approx = frac;
    end
  endfunction

  task automatic rms_norm(
      input  int layer,
      input  logic signed [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out_vec[0:D_MODEL-1],
      input  logic signed [DATA_W-1:0] weight [0:N_LAYER*D_MODEL-1]
  );
    integer i;
    real sum_sq;
    real inv_rms;
    real v;
    begin
      sum_sq = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        v = q_to_real(in_vec[i]);
        sum_sq = sum_sq + v * v;
      end
      inv_rms = 1.0 / $sqrt((sum_sq / D_MODEL) + RMS_EPS);
      for (i = 0; i < D_MODEL; i = i + 1) begin
        v = q_to_real(in_vec[i]) * inv_rms;
        out_vec[i] = mul_q(real_to_q(v), weight[layer * D_MODEL + i]);
      end
    end
  endtask

  task automatic rms_norm_out(
      input  logic signed [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out_vec[0:D_MODEL-1]
  );
    integer i;
    real sum_sq;
    real inv_rms;
    real v;
    begin
      sum_sq = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        v = q_to_real(in_vec[i]);
        sum_sq = sum_sq + v * v;
      end
      inv_rms = 1.0 / $sqrt((sum_sq / D_MODEL) + RMS_EPS);
      for (i = 0; i < D_MODEL; i = i + 1) begin
        v = q_to_real(in_vec[i]) * inv_rms;
        out_vec[i] = mul_q(real_to_q(v), out_norm_w[i]);
      end
    end
  endtask

  task automatic apply_rope(
      input int pos,
      inout logic signed [DATA_W-1:0] q_vec [0:D_MODEL-1],
      inout logic signed [DATA_W-1:0] k_vec [0:D_KV-1]
  );
    integer h;
    integer i;
    integer head_dim2;
    real angle;
    real freq;
    real cos_t;
    real sin_t;
    real q0;
    real q1;
    real k0;
    real k1;
    begin
      head_dim2 = ROPE_DIM / 2;
      for (h = 0; h < N_HEAD; h = h + 1) begin
        for (i = 0; i < head_dim2; i = i + 1) begin
          freq = 1.0 / (ROPE_BASE ** (2.0 * i / ROPE_DIM));
          angle = pos * freq;
          cos_t = $cos(angle);
          sin_t = $sin(angle);
          q0 = q_to_real(q_vec[h * HEAD_DIM + 2*i]);
          q1 = q_to_real(q_vec[h * HEAD_DIM + 2*i + 1]);
          q_vec[h * HEAD_DIM + 2*i]     = real_to_q(q0 * cos_t - q1 * sin_t);
          q_vec[h * HEAD_DIM + 2*i + 1] = real_to_q(q0 * sin_t + q1 * cos_t);
        end
      end
      for (h = 0; h < N_KV_HEAD; h = h + 1) begin
        for (i = 0; i < head_dim2; i = i + 1) begin
          freq = 1.0 / (ROPE_BASE ** (2.0 * i / ROPE_DIM));
          angle = pos * freq;
          cos_t = $cos(angle);
          sin_t = $sin(angle);
          k0 = q_to_real(k_vec[h * HEAD_DIM + 2*i]);
          k1 = q_to_real(k_vec[h * HEAD_DIM + 2*i + 1]);
          k_vec[h * HEAD_DIM + 2*i]     = real_to_q(k0 * cos_t - k1 * sin_t);
          k_vec[h * HEAD_DIM + 2*i + 1] = real_to_q(k0 * sin_t + k1 * cos_t);
        end
      end
    end
  endtask

  task automatic matvec_q(
      input  int layer,
      input  logic signed [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out [0:D_MODEL-1]
  );
    integer i;
    integer j;
    logic signed [ACC_W-1:0] acc;
    begin
      for (j = 0; j < D_MODEL; j = j + 1) begin
        acc = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          acc = acc + ((vec[i] * attn_q_w[layer*D_MODEL*D_MODEL + i*D_MODEL + j]) >>> FRAC_W);
        end
        out[j] = sat_q(acc);
      end
    end
  endtask

  task automatic matvec_k(
      input  int layer,
      input  logic signed [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out [0:D_KV-1]
  );
    integer i;
    integer j;
    logic signed [ACC_W-1:0] acc;
    begin
      for (j = 0; j < D_KV; j = j + 1) begin
        acc = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          acc = acc + ((vec[i] * attn_k_w[layer*D_MODEL*D_KV + i*D_KV + j]) >>> FRAC_W);
        end
        out[j] = sat_q(acc);
      end
    end
  endtask

  task automatic matvec_v(
      input  int layer,
      input  logic signed [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out [0:D_KV-1]
  );
    integer i;
    integer j;
    logic signed [ACC_W-1:0] acc;
    begin
      for (j = 0; j < D_KV; j = j + 1) begin
        acc = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          acc = acc + ((vec[i] * attn_v_w[layer*D_MODEL*D_KV + i*D_KV + j]) >>> FRAC_W);
        end
        out[j] = sat_q(acc);
      end
    end
  endtask

  task automatic matvec_attn_out(
      input  int layer,
      input  logic signed [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out [0:D_MODEL-1]
  );
    integer i;
    integer j;
    logic signed [ACC_W-1:0] acc;
    begin
      for (j = 0; j < D_MODEL; j = j + 1) begin
        acc = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          acc = acc + ((vec[i] * attn_out_w[layer*D_MODEL*D_MODEL + i*D_MODEL + j]) >>> FRAC_W);
        end
        out[j] = sat_q(acc);
      end
    end
  endtask

  task automatic matvec_ffn_gate(
      input  int layer,
      input  logic signed [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out [0:D_FF-1]
  );
    integer i;
    integer j;
    logic signed [ACC_W-1:0] acc;
    begin
      for (j = 0; j < D_FF; j = j + 1) begin
        acc = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          acc = acc + ((vec[i] * ffn_gate_w[layer*D_MODEL*D_FF + i*D_FF + j]) >>> FRAC_W);
        end
        out[j] = sat_q(acc);
      end
    end
  endtask

  task automatic matvec_ffn_up(
      input  int layer,
      input  logic signed [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out [0:D_FF-1]
  );
    integer i;
    integer j;
    logic signed [ACC_W-1:0] acc;
    begin
      for (j = 0; j < D_FF; j = j + 1) begin
        acc = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          acc = acc + ((vec[i] * ffn_up_w[layer*D_MODEL*D_FF + i*D_FF + j]) >>> FRAC_W);
        end
        out[j] = sat_q(acc);
      end
    end
  endtask

  task automatic matvec_ffn_dn(
      input  int layer,
      input  logic signed [DATA_W-1:0] vec [0:D_FF-1],
      output logic signed [DATA_W-1:0] out [0:D_MODEL-1]
  );
    integer i;
    integer j;
    logic signed [ACC_W-1:0] acc;
    begin
      for (j = 0; j < D_MODEL; j = j + 1) begin
        acc = '0;
        for (i = 0; i < D_FF; i = i + 1) begin
          acc = acc + ((vec[i] * ffn_dn_w[layer*D_FF*D_MODEL + i*D_MODEL + j]) >>> FRAC_W);
        end
        out[j] = sat_q(acc);
      end
    end
  endtask

  task automatic attention(
      input  int layer,
      input  int pos,
      input  logic signed [DATA_W-1:0] q_vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out [0:D_MODEL-1]
  );
    integer h;
    integer t;
    integer d;
    integer i;
    integer kv_head;
    logic signed [ACC_W-1:0] acc;
    logic signed [DATA_W-1:0] scores [0:MAX_CTX-1];
    logic signed [DATA_W-1:0] weights[0:MAX_CTX-1];
    logic signed [ACC_W-1:0] sum_exp;
    real scale;
    logic signed [DATA_W-1:0] score_q;
    begin
      scale = 1.0 / $sqrt(HEAD_DIM);
      for (h = 0; h < N_HEAD; h = h + 1) begin
        kv_head = h / (N_HEAD / N_KV_HEAD);
        for (t = 0; t <= pos; t = t + 1) begin
          acc = '0;
          for (d = 0; d < HEAD_DIM; d = d + 1) begin
            i = h * HEAD_DIM + d;
            acc = acc + (q_vec[i] * k_cache[layer][idx2d(t, kv_head * HEAD_DIM + d, D_KV)]);
          end
          acc = acc >>> FRAC_W;
          score_q = sat_q(acc);
          scores[t] = real_to_q(q_to_real(score_q) * scale);
        end
        sum_exp = '0;
        for (t = 0; t <= pos; t = t + 1) begin
          weights[t] = exp_approx(scores[t]);
          sum_exp = sum_exp + weights[t];
        end
        for (t = 0; t <= pos; t = t + 1) begin
          weights[t] = div_q(weights[t], sum_exp[DATA_W-1:0]);
        end
        for (d = 0; d < HEAD_DIM; d = d + 1) begin
          acc = '0;
          for (t = 0; t <= pos; t = t + 1) begin
            acc = acc + ((weights[t] *
                          v_cache[layer][idx2d(t, kv_head * HEAD_DIM + d, D_KV)]) >>> FRAC_W);
          end
          out[h * HEAD_DIM + d] = sat_q(acc);
        end
      end
    end
  endtask

  task automatic compute_logits(
      input  logic signed [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic [31:0] token_id
  );
    integer v;
    integer d;
    logic signed [ACC_W-1:0] acc;
    logic signed [ACC_W-1:0] best_acc;
    logic signed [ACC_W-1:0] top_score [0:4];
    logic [31:0] top_id [0:4];
    integer k;
    integer insert;
    begin
      best_acc = -32'sh4000_0000;
      token_id = 0;
      for (k = 0; k < 5; k = k + 1) begin
        top_score[k] = -32'sh4000_0000;
        top_id[k] = 0;
      end
      for (v = 0; v < VOCAB; v = v + 1) begin
        acc = '0;
        for (d = 0; d < D_MODEL; d = d + 1) begin
          acc = acc + ((vec[d] * out_weight[idx2d(v, d, D_MODEL)]) >>> FRAC_W);
        end
        if (acc > best_acc) begin
          best_acc = acc;
          token_id = v[31:0];
        end
        if (acc > top_score[4]) begin
          insert = 4;
          while (insert > 0 && acc > top_score[insert-1]) begin
            top_score[insert] = top_score[insert-1];
            top_id[insert] = top_id[insert-1];
            insert = insert - 1;
          end
          top_score[insert] = acc;
          top_id[insert] = v[31:0];
        end
      end
      for (k = 0; k < 5; k = k + 1) begin
        debug_topk_id[k] = top_id[k];
        debug_topk_score[k] = top_score[k];
      end
    end
  endtask

  integer pos;
  integer layer;
  integer i;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done <= 1'b0;
      next_token_id <= 0;
    end else begin
      done <= 1'b0;
      if (start) begin
        for (layer = 0; layer < N_LAYER; layer = layer + 1) begin
          for (pos = 0; pos < MAX_CTX; pos = pos + 1) begin
            for (i = 0; i < D_KV; i = i + 1) begin
              k_cache[layer][idx2d(pos, i, D_KV)] <= '0;
              v_cache[layer][idx2d(pos, i, D_KV)] <= '0;
            end
          end
        end

        for (pos = 0; pos < prompt_len; pos = pos + 1) begin
          for (i = 0; i < D_MODEL; i = i + 1) begin
            x[i] = token_embd[idx2d(input_tokens[pos], i, D_MODEL)];
          end

          for (layer = 0; layer < N_LAYER; layer = layer + 1) begin
            rms_norm(layer, x, x_norm, attn_norm_w);
            matvec_q(layer, x_norm, q);
            matvec_k(layer, x_norm, k);
            matvec_v(layer, x_norm, v);
            apply_rope(pos, q, k);
            for (i = 0; i < D_KV; i = i + 1) begin
              k_cache[layer][idx2d(pos, i, D_KV)] = k[i];
              v_cache[layer][idx2d(pos, i, D_KV)] = v[i];
            end
            attention(layer, pos, q, attn_out);
            matvec_attn_out(layer, attn_out, attn_proj);
            for (i = 0; i < D_MODEL; i = i + 1)
              x[i] = add_q(x[i], attn_proj[i]);

            rms_norm(layer, x, x_norm, ffn_norm_w);
            matvec_ffn_gate(layer, x_norm, ffn_gate);
            matvec_ffn_up(layer, x_norm, ffn_up);
            for (i = 0; i < D_FF; i = i + 1) begin
              ffn_act[i] = mul_q(silu_approx(ffn_gate[i]), ffn_up[i]);
            end
            matvec_ffn_dn(layer, ffn_act, ffn_out);
            for (i = 0; i < D_MODEL; i = i + 1)
              x[i] = add_q(x[i], ffn_out[i]);
          end
        end

        rms_norm_out(x, x_norm);
        compute_logits(x_norm, next_token_id);
        done <= 1'b1;
      end
    end
  end
endmodule
