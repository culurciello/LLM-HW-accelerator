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
    parameter real RMS_EPS = 1e-5,
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
  localparam int D_KV = N_KV_HEAD * HEAD_DIM;

  logic [DATA_W-1:0] token_embd_fp16 [0:VOCAB*D_MODEL-1];
  logic [DATA_W-1:0] out_weight_fp16 [0:VOCAB*D_MODEL-1];
  logic [DATA_W-1:0] out_norm_w_fp16 [0:D_MODEL-1];

  logic [DATA_W-1:0] attn_norm_w_fp16 [0:N_LAYER*D_MODEL-1];
  logic [DATA_W-1:0] ffn_norm_w_fp16  [0:N_LAYER*D_MODEL-1];

  logic [DATA_W-1:0] attn_q_w_fp16    [0:N_LAYER*D_MODEL*D_MODEL-1];
  logic [DATA_W-1:0] attn_k_w_fp16    [0:N_LAYER*D_MODEL*D_KV-1];
  logic [DATA_W-1:0] attn_v_w_fp16    [0:N_LAYER*D_MODEL*D_KV-1];
  logic [DATA_W-1:0] attn_out_w_fp16  [0:N_LAYER*D_MODEL*D_MODEL-1];

  logic [DATA_W-1:0] ffn_gate_w_fp16  [0:N_LAYER*D_MODEL*D_FF-1];
  logic [DATA_W-1:0] ffn_up_w_fp16    [0:N_LAYER*D_MODEL*D_FF-1];
  logic [DATA_W-1:0] ffn_dn_w_fp16    [0:N_LAYER*D_FF*D_MODEL-1];

  logic [DATA_W-1:0] k_cache_fp16 [0:N_LAYER-1][0:MAX_CTX*D_KV-1];
  logic [DATA_W-1:0] v_cache_fp16 [0:N_LAYER-1][0:MAX_CTX*D_KV-1];

  logic [31:0] input_tokens [0:MAX_CTX-1];
  logic [DATA_W-1:0] x_fp16        [0:D_MODEL-1];
  logic [DATA_W-1:0] x_norm_fp16   [0:D_MODEL-1];
  logic [DATA_W-1:0] q_fp16        [0:D_MODEL-1];
  logic [DATA_W-1:0] k_fp16        [0:D_KV-1];
  logic [DATA_W-1:0] v_fp16        [0:D_KV-1];
  logic [DATA_W-1:0] attn_out_fp16 [0:D_MODEL-1];
  logic [DATA_W-1:0] attn_proj_fp16[0:D_MODEL-1];
  logic [DATA_W-1:0] ffn_gate_fp16 [0:D_FF-1];
  logic [DATA_W-1:0] ffn_up_fp16   [0:D_FF-1];
  logic [DATA_W-1:0] ffn_act_fp16  [0:D_FF-1];
  logic [DATA_W-1:0] ffn_out_fp16  [0:D_MODEL-1];

  logic [31:0] debug_topk_id [0:TOP_K-1];
  logic signed [ACC_W-1:0] debug_topk_score [0:TOP_K-1];

  string dump_dir;
  integer dump_layer_arg;
  integer dump_all;
  integer dump_pos_arg;
  integer dump_pos;
  bit dump_enable;

  initial begin
    dump_enable = $value$plusargs("dump_dir=%s", dump_dir);
    if (!$value$plusargs("dump_layer=%d", dump_layer_arg))
      dump_layer_arg = 0;
    if (!$value$plusargs("dump_all=%d", dump_all))
      dump_all = 0;
    if (!$value$plusargs("dump_pos=%d", dump_pos_arg))
      dump_pos_arg = -1;
  end

  function automatic int idx2d(input int row, input int col, input int stride);
    begin
      idx2d = row * stride + col;
    end
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

  function automatic logic [15:0] silu_fp16(
      input logic [15:0] x_in
  );
    real val_r;
    real res_r;
    begin
      val_r = fp16_to_real(x_in);
      res_r = val_r / (1.0 + $exp(-val_r));
      silu_fp16 = real_to_fp16(res_r);
    end
  endfunction

  task automatic rms_norm_fp16(
      input  int layer,
      input  logic [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out_vec[0:D_MODEL-1],
      input  logic [DATA_W-1:0] weight [0:N_LAYER*D_MODEL-1]
  );
    integer i;
    real sum_sq;
    real inv_rms;
    real val;
    logic [15:0] val_fp16;
    begin
      sum_sq = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = fp16_to_real(in_vec[i]);
        sum_sq = sum_sq + val * val;
      end
      inv_rms = 1.0 / $sqrt((sum_sq / D_MODEL) + RMS_EPS);
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = fp16_to_real(in_vec[i]) * inv_rms;
        val_fp16 = real_to_fp16(val);
        out_vec[i] = fp16_mul(val_fp16, weight[layer * D_MODEL + i]);
      end
    end
  endtask

  task automatic dump_vec_fp16(
      input string tag,
      input int layer_id,
      input int len,
      input logic [DATA_W-1:0] vec []
  );
    integer fd;
    integer idx;
    string path;
    logic [DATA_W-1:0] tmp;
    begin
      if (!dump_enable)
        return;
      path = $sformatf("%s/layer%0d_%s.mem", dump_dir, layer_id, tag);
      fd = $fopen(path, "w");
      if (fd == 0) begin
        $display("dump open failed: %s", path);
      end else begin
        for (idx = 0; idx < len; idx = idx + 1) begin
          tmp = vec[idx];
          $fdisplay(fd, "%04x", tmp);
        end
        $fclose(fd);
      end
    end
  endtask

  task automatic rms_norm_out_fp16(
      input  logic [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out_vec[0:D_MODEL-1]
  );
    integer i;
    real sum_sq;
    real inv_rms;
    real val;
    logic [15:0] val_fp16;
    begin
      sum_sq = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = fp16_to_real(in_vec[i]);
        sum_sq = sum_sq + val * val;
      end
      inv_rms = 1.0 / $sqrt((sum_sq / D_MODEL) + RMS_EPS);
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = fp16_to_real(in_vec[i]) * inv_rms;
        val_fp16 = real_to_fp16(val);
        out_vec[i] = fp16_mul(val_fp16, out_norm_w_fp16[i]);
      end
    end
  endtask

  task automatic apply_rope_fp16(
      input int pos,
      inout logic [DATA_W-1:0] q_vec [0:D_MODEL-1],
      inout logic [DATA_W-1:0] k_vec [0:D_KV-1]
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
          q0 = fp16_to_real(q_vec[h * HEAD_DIM + 2*i]);
          q1 = fp16_to_real(q_vec[h * HEAD_DIM + 2*i + 1]);
          q_vec[h * HEAD_DIM + 2*i]     = real_to_fp16(q0 * cos_t - q1 * sin_t);
          q_vec[h * HEAD_DIM + 2*i + 1] = real_to_fp16(q0 * sin_t + q1 * cos_t);
        end
      end
      for (h = 0; h < N_KV_HEAD; h = h + 1) begin
        for (i = 0; i < head_dim2; i = i + 1) begin
          freq = 1.0 / (ROPE_BASE ** (2.0 * i / ROPE_DIM));
          angle = pos * freq;
          cos_t = $cos(angle);
          sin_t = $sin(angle);
          k0 = fp16_to_real(k_vec[h * HEAD_DIM + 2*i]);
          k1 = fp16_to_real(k_vec[h * HEAD_DIM + 2*i + 1]);
          k_vec[h * HEAD_DIM + 2*i]     = real_to_fp16(k0 * cos_t - k1 * sin_t);
          k_vec[h * HEAD_DIM + 2*i + 1] = real_to_fp16(k0 * sin_t + k1 * cos_t);
        end
      end
    end
  endtask

  task automatic matvec_q_fp16(
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
          prod_fp16 = fp16_mul(vec[i], attn_q_w_fp16[layer*D_MODEL*D_MODEL + i*D_MODEL + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = acc_fp16;
      end
    end
  endtask

  task automatic matvec_k_fp16(
      input  int layer,
      input  logic [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out [0:D_KV-1]
  );
    integer i;
    integer j;
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_KV; j = j + 1) begin
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          prod_fp16 = fp16_mul(vec[i], attn_k_w_fp16[layer*D_MODEL*D_KV + i*D_KV + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = acc_fp16;
      end
    end
  endtask

  task automatic matvec_v_fp16(
      input  int layer,
      input  logic [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out [0:D_KV-1]
  );
    integer i;
    integer j;
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_KV; j = j + 1) begin
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          prod_fp16 = fp16_mul(vec[i], attn_v_w_fp16[layer*D_MODEL*D_KV + i*D_KV + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = acc_fp16;
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
          prod_fp16 = fp16_mul(vec[i], attn_out_w_fp16[layer*D_MODEL*D_MODEL + i*D_MODEL + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = acc_fp16;
      end
    end
  endtask

  task automatic matvec_ffn_gate_fp16(
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
          prod_fp16 = fp16_mul(vec[i], ffn_gate_w_fp16[layer*D_MODEL*D_FF + i*D_FF + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = acc_fp16;
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
          prod_fp16 = fp16_mul(vec[i], ffn_up_w_fp16[layer*D_MODEL*D_FF + i*D_FF + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = acc_fp16;
      end
    end
  endtask

  task automatic matvec_ffn_dn_fp16(
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
          prod_fp16 = fp16_mul(vec[i], ffn_dn_w_fp16[layer*D_FF*D_MODEL + i*D_MODEL + j]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        out[j] = acc_fp16;
      end
    end
  endtask

  task automatic attention_fp16(
      input  int layer,
      input  int pos,
      input  logic [DATA_W-1:0] q_vec [0:D_MODEL-1],
      output logic [DATA_W-1:0] out [0:D_MODEL-1]
  );
    integer h;
    integer t;
    integer d;
    integer i;
    integer kv_head;
    real scores_r [0:MAX_CTX-1];
    real weight_r [0:MAX_CTX-1];
    real max_score;
    real sum_exp;
    logic [DATA_W-1:0] weight_fp16 [0:MAX_CTX-1];
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    real scale;
    begin
      scale = 1.0 / $sqrt(HEAD_DIM);
      for (h = 0; h < N_HEAD; h = h + 1) begin
        kv_head = h / (N_HEAD / N_KV_HEAD);
        for (t = 0; t <= pos; t = t + 1) begin
          acc_fp16 = '0;
          for (d = 0; d < HEAD_DIM; d = d + 1) begin
            i = h * HEAD_DIM + d;
            prod_fp16 = fp16_mul(q_vec[i],
                                 k_cache_fp16[layer][idx2d(t, kv_head * HEAD_DIM + d, D_KV)]);
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
        for (t = 0; t <= pos; t = t + 1) begin
          weight_fp16[t] = real_to_fp16(weight_r[t] / sum_exp);
        end
        for (d = 0; d < HEAD_DIM; d = d + 1) begin
          acc_fp16 = '0;
          for (t = 0; t <= pos; t = t + 1) begin
            prod_fp16 = fp16_mul(weight_fp16[t],
                                 v_cache_fp16[layer][idx2d(t, kv_head * HEAD_DIM + d, D_KV)]);
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
    real best_r;
    real top_r [0:TOP_K-1];
    logic [31:0] top_id [0:TOP_K-1];
    logic [15:0] score_fp16;
    integer k_idx;
    integer insert;
    begin
      best_r = -1.0e30;
      token_id = 0;
      for (k_idx = 0; k_idx < TOP_K; k_idx = k_idx + 1) begin
        top_r[k_idx] = -1.0e30;
        top_id[k_idx] = 0;
      end
      for (tok = 0; tok < VOCAB; tok = tok + 1) begin
        acc_fp16 = '0;
        for (d = 0; d < D_MODEL; d = d + 1) begin
          prod_fp16 = fp16_mul(vec[d], out_weight_fp16[idx2d(tok, d, D_MODEL)]);
          acc_fp16 = fp16_add(acc_fp16, prod_fp16);
        end
        acc_r = fp16_to_real(acc_fp16);
        if (acc_r > best_r) begin
          best_r = acc_r;
          token_id = tok[31:0];
        end
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
  bit dump_this;

  /* verilator lint_off BLKSEQ */
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done <= 1'b0;
      next_token_id <= 0;
    end else begin
      done <= 1'b0;
      if (start) begin
        if (dump_pos_arg < 0) begin
          if (prompt_len == 0)
            dump_pos = 0;
          else
            dump_pos = prompt_len - 1;
        end else begin
          dump_pos = dump_pos_arg;
        end

        for (layer = 0; layer < N_LAYER; layer = layer + 1) begin
          for (pos = 0; pos < MAX_CTX; pos = pos + 1) begin
            for (i = 0; i < D_KV; i = i + 1) begin
              k_cache_fp16[layer][idx2d(pos, i, D_KV)] <= '0;
              v_cache_fp16[layer][idx2d(pos, i, D_KV)] <= '0;
            end
          end
        end

        for (pos = 0; pos < prompt_len; pos = pos + 1) begin
          for (i = 0; i < D_MODEL; i = i + 1) begin
            x_fp16[i] = token_embd_fp16[idx2d(input_tokens[pos], i, D_MODEL)];
          end
          for (layer = 0; layer < N_LAYER; layer = layer + 1) begin
            dump_this = dump_enable && (pos == dump_pos) &&
                        ((dump_all != 0) || (layer == dump_layer_arg));

            if (dump_this)
              dump_vec_fp16("x_in", layer, D_MODEL, x_fp16);
            rms_norm_fp16(layer, x_fp16, x_norm_fp16, attn_norm_w_fp16);
            if (dump_this)
              dump_vec_fp16("attn_norm", layer, D_MODEL, x_norm_fp16);
            matvec_q_fp16(layer, x_norm_fp16, q_fp16);
            matvec_k_fp16(layer, x_norm_fp16, k_fp16);
            matvec_v_fp16(layer, x_norm_fp16, v_fp16);
            apply_rope_fp16(pos, q_fp16, k_fp16);
            if (dump_this) begin
              dump_vec_fp16("q", layer, D_MODEL, q_fp16);
              dump_vec_fp16("k", layer, D_KV, k_fp16);
              dump_vec_fp16("v", layer, D_KV, v_fp16);
            end
            for (i = 0; i < D_KV; i = i + 1) begin
              k_cache_fp16[layer][idx2d(pos, i, D_KV)] = k_fp16[i];
              v_cache_fp16[layer][idx2d(pos, i, D_KV)] = v_fp16[i];
            end
            attention_fp16(layer, pos, q_fp16, attn_out_fp16);
            if (dump_this)
              dump_vec_fp16("attn_out", layer, D_MODEL, attn_out_fp16);
            matvec_attn_out_fp16(layer, attn_out_fp16, attn_proj_fp16);
            if (dump_this)
              dump_vec_fp16("attn_proj", layer, D_MODEL, attn_proj_fp16);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              x_fp16[i] = fp16_add(x_fp16[i], attn_proj_fp16[i]);
            end
            if (dump_this)
              dump_vec_fp16("x_attn", layer, D_MODEL, x_fp16);

            rms_norm_fp16(layer, x_fp16, x_norm_fp16, ffn_norm_w_fp16);
            if (dump_this)
              dump_vec_fp16("ffn_norm", layer, D_MODEL, x_norm_fp16);
            matvec_ffn_gate_fp16(layer, x_norm_fp16, ffn_gate_fp16);
            matvec_ffn_up_fp16(layer, x_norm_fp16, ffn_up_fp16);
            if (dump_this) begin
              dump_vec_fp16("ffn_gate", layer, D_FF, ffn_gate_fp16);
              dump_vec_fp16("ffn_up", layer, D_FF, ffn_up_fp16);
            end
            for (i = 0; i < D_FF; i = i + 1) begin
              ffn_act_fp16[i] = fp16_mul(silu_fp16(ffn_gate_fp16[i]), ffn_up_fp16[i]);
            end
            if (dump_this)
              dump_vec_fp16("ffn_act", layer, D_FF, ffn_act_fp16);
            matvec_ffn_dn_fp16(layer, ffn_act_fp16, ffn_out_fp16);
            if (dump_this)
              dump_vec_fp16("ffn_out", layer, D_MODEL, ffn_out_fp16);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              x_fp16[i] = fp16_add(x_fp16[i], ffn_out_fp16[i]);
            end
            if (dump_this)
              dump_vec_fp16("x_out", layer, D_MODEL, x_fp16);
          end
        end

        rms_norm_out_fp16(x_fp16, x_norm_fp16);
        compute_logits_fp16(x_norm_fp16, next_token_id);
        done <= 1'b1;
      end
    end
  end
  /* verilator lint_on BLKSEQ */
endmodule
