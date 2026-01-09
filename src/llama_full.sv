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

  string dump_dir;
  integer use_fp16_arg;
  bit use_fp16;
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
    if ($value$plusargs("use_fp16=%d", use_fp16_arg))
      use_fp16 = (use_fp16_arg != 0);
    else
      use_fp16 = 1'b0;
  end

  function automatic int idx2d(input int row, input int col, input int stride);
    begin
      idx2d = row * stride + col;
    end
  endfunction

  function automatic real q_to_real(input logic signed [DATA_W-1:0] qv);
    begin
      q_to_real = qv / real'(1 << FRAC_W);
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] real_to_q(input real val);
    integer tmp;
    begin
      tmp = $rtoi(val * (1 << FRAC_W));
      real_to_q = sat_q(tmp);
    end
  endfunction

  function automatic logic [15:0] q_to_fp16(input logic signed [DATA_W-1:0] qv);
    real v;
    integer sign;
    integer exp;
    real mant;
    integer man_bits;
    begin
      v = q_to_real(qv);
      if (v == 0.0) begin
        q_to_fp16 = 16'h0000;
      end else begin
        sign = (v < 0.0);
        if (v < 0.0)
          v = -v;
        exp = $rtoi($floor($ln(v) / $ln(2.0)));
        mant = v / (2.0 ** exp);
        man_bits = $rtoi((mant - 1.0) * 1024.0 + 0.5);
        if (man_bits < 0)
          man_bits = 0;
        if (man_bits > 1023)
          man_bits = 1023;
        exp = exp + 15;
        if (exp <= 0) begin
          q_to_fp16 = 16'h0000;
        end else if (exp >= 31) begin
          q_to_fp16 = {sign[0], 5'h1F, 10'h000};
        end else begin
          q_to_fp16 = {sign[0], exp[4:0], man_bits[9:0]};
        end
      end
    end
  endfunction

  function automatic logic [15:0] real_to_fp16(input real v);
    integer sign;
    integer exp;
    real mant;
    integer man_bits;
    begin
      if (v == 0.0) begin
        real_to_fp16 = 16'h0000;
      end else begin
        sign = (v < 0.0);
        if (v < 0.0)
          v = -v;
        exp = $rtoi($floor($ln(v) / $ln(2.0)));
        mant = v / (2.0 ** exp);
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

  function automatic logic signed [DATA_W-1:0] fp16_to_q(input logic [15:0] h);
    real v;
    integer sign;
    integer exp;
    integer man;
    real mant;
    begin
      if (h[14:0] == 0) begin
        fp16_to_q = '0;
      end else begin
        sign = h[15];
        exp = h[14:10] - 15;
        man = h[9:0];
        mant = 1.0 + (man / 1024.0);
        v = mant * (2.0 ** exp);
        if (sign)
          v = -v;
        fp16_to_q = real_to_q(v);
      end
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
    real val;
    logic [15:0] inv_fp16;
    logic [15:0] mul_fp16;
    logic [15:0] val_fp16;
    begin
      sum_sq = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = use_fp16 ? fp16_to_real(in_vec[i]) : q_to_real(in_vec[i]);
        sum_sq = sum_sq + val * val;
      end
      inv_rms = 1.0 / $sqrt((sum_sq / D_MODEL) + RMS_EPS);
      inv_fp16 = real_to_fp16(inv_rms);
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = (use_fp16 ? fp16_to_real(in_vec[i]) : q_to_real(in_vec[i])) * inv_rms;
        val_fp16 = real_to_fp16(val);
        if (use_fp16) begin
          mul_fp16 = fp16_mul(val_fp16, weight[layer * D_MODEL + i]);
          out_vec[i] = mul_fp16;
        end else begin
          mul_fp16 = fp16_mul(val_fp16, q_to_fp16(weight[layer * D_MODEL + i]));
          out_vec[i] = fp16_to_q(mul_fp16);
        end
      end
    end
  endtask

  task automatic dump_vec(
      input string tag,
      input int layer_id,
      input int len,
      input logic signed [DATA_W-1:0] vec []
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

  task automatic rms_norm_out(
      input  logic signed [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out_vec[0:D_MODEL-1]
  );
    integer i;
    real sum_sq;
    real inv_rms;
    real val;
    logic [15:0] inv_fp16;
    logic [15:0] mul_fp16;
    logic [15:0] val_fp16;
    begin
      sum_sq = 0.0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = use_fp16 ? fp16_to_real(in_vec[i]) : q_to_real(in_vec[i]);
        sum_sq = sum_sq + val * val;
      end
      inv_rms = 1.0 / $sqrt((sum_sq / D_MODEL) + RMS_EPS);
      inv_fp16 = real_to_fp16(inv_rms);
      for (i = 0; i < D_MODEL; i = i + 1) begin
        val = (use_fp16 ? fp16_to_real(in_vec[i]) : q_to_real(in_vec[i])) * inv_rms;
        val_fp16 = real_to_fp16(val);
        if (use_fp16) begin
          mul_fp16 = fp16_mul(val_fp16, out_norm_w[i]);
          out_vec[i] = mul_fp16;
        end else begin
          mul_fp16 = fp16_mul(val_fp16, q_to_fp16(out_norm_w[i]));
          out_vec[i] = fp16_to_q(mul_fp16);
        end
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
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_MODEL; j = j + 1) begin
        acc = '0;
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          if (use_fp16) begin
            prod_fp16 = fp16_mul(vec[i], attn_q_w[layer*D_MODEL*D_MODEL + i*D_MODEL + j]);
            acc_fp16 = fp16_add(acc_fp16, prod_fp16);
          end else begin
            acc = acc + ((vec[i] * attn_q_w[layer*D_MODEL*D_MODEL + i*D_MODEL + j]) >>> FRAC_W);
          end
        end
        out[j] = use_fp16 ? acc_fp16 : sat_q(acc);
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
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_KV; j = j + 1) begin
        acc = '0;
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          if (use_fp16) begin
            prod_fp16 = fp16_mul(vec[i], attn_k_w[layer*D_MODEL*D_KV + i*D_KV + j]);
            acc_fp16 = fp16_add(acc_fp16, prod_fp16);
          end else begin
            acc = acc + ((vec[i] * attn_k_w[layer*D_MODEL*D_KV + i*D_KV + j]) >>> FRAC_W);
          end
        end
        out[j] = use_fp16 ? acc_fp16 : sat_q(acc);
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
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_KV; j = j + 1) begin
        acc = '0;
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          if (use_fp16) begin
            prod_fp16 = fp16_mul(vec[i], attn_v_w[layer*D_MODEL*D_KV + i*D_KV + j]);
            acc_fp16 = fp16_add(acc_fp16, prod_fp16);
          end else begin
            acc = acc + ((vec[i] * attn_v_w[layer*D_MODEL*D_KV + i*D_KV + j]) >>> FRAC_W);
          end
        end
        out[j] = use_fp16 ? acc_fp16 : sat_q(acc);
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
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_MODEL; j = j + 1) begin
        acc = '0;
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          if (use_fp16) begin
            prod_fp16 = fp16_mul(vec[i], attn_out_w[layer*D_MODEL*D_MODEL + i*D_MODEL + j]);
            acc_fp16 = fp16_add(acc_fp16, prod_fp16);
          end else begin
            acc = acc + ((vec[i] * attn_out_w[layer*D_MODEL*D_MODEL + i*D_MODEL + j]) >>> FRAC_W);
          end
        end
        out[j] = use_fp16 ? acc_fp16 : sat_q(acc);
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
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_FF; j = j + 1) begin
        acc = '0;
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          if (use_fp16) begin
            prod_fp16 = fp16_mul(vec[i], ffn_gate_w[layer*D_MODEL*D_FF + i*D_FF + j]);
            acc_fp16 = fp16_add(acc_fp16, prod_fp16);
          end else begin
            acc = acc + ((vec[i] * ffn_gate_w[layer*D_MODEL*D_FF + i*D_FF + j]) >>> FRAC_W);
          end
        end
        out[j] = use_fp16 ? acc_fp16 : sat_q(acc);
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
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_FF; j = j + 1) begin
        acc = '0;
        acc_fp16 = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          if (use_fp16) begin
            prod_fp16 = fp16_mul(vec[i], ffn_up_w[layer*D_MODEL*D_FF + i*D_FF + j]);
            acc_fp16 = fp16_add(acc_fp16, prod_fp16);
          end else begin
            acc = acc + ((vec[i] * ffn_up_w[layer*D_MODEL*D_FF + i*D_FF + j]) >>> FRAC_W);
          end
        end
        out[j] = use_fp16 ? acc_fp16 : sat_q(acc);
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
    logic [DATA_W-1:0] acc_fp16;
    logic [DATA_W-1:0] prod_fp16;
    begin
      for (j = 0; j < D_MODEL; j = j + 1) begin
        acc = '0;
        acc_fp16 = '0;
        for (i = 0; i < D_FF; i = i + 1) begin
          if (use_fp16) begin
            prod_fp16 = fp16_mul(vec[i], ffn_dn_w[layer*D_FF*D_MODEL + i*D_MODEL + j]);
            acc_fp16 = fp16_add(acc_fp16, prod_fp16);
          end else begin
            acc = acc + ((vec[i] * ffn_dn_w[layer*D_FF*D_MODEL + i*D_MODEL + j]) >>> FRAC_W);
          end
        end
        out[j] = use_fp16 ? acc_fp16 : sat_q(acc);
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
    real score_r;
    real weight_r[0:MAX_CTX-1];
    begin
      scale = 1.0 / $sqrt(HEAD_DIM);
      for (h = 0; h < N_HEAD; h = h + 1) begin
        kv_head = h / (N_HEAD / N_KV_HEAD);
        for (t = 0; t <= pos; t = t + 1) begin
          acc = '0;
          for (d = 0; d < HEAD_DIM; d = d + 1) begin
            i = h * HEAD_DIM + d;
            if (use_fp16) begin
              acc = acc + $signed(fp16_to_q(fp16_mul(q_vec[i],
                                                   k_cache[layer][idx2d(t, kv_head * HEAD_DIM + d, D_KV)])));
            end else begin
              acc = acc + (q_vec[i] * k_cache[layer][idx2d(t, kv_head * HEAD_DIM + d, D_KV)]);
            end
          end
          acc = acc >>> FRAC_W;
          score_q = sat_q(acc);
          score_r = (use_fp16 ? fp16_to_real(score_q) : q_to_real(score_q)) * scale;
          scores[t] = real_to_q(score_r);
        end
        sum_exp = '0;
        if (use_fp16) begin
          real max_score;
          max_score = -1.0e30;
          for (t = 0; t <= pos; t = t + 1) begin
            if (fp16_to_real(scores[t]) > max_score)
              max_score = fp16_to_real(scores[t]);
          end
          for (t = 0; t <= pos; t = t + 1) begin
            weight_r[t] = $exp(fp16_to_real(scores[t]) - max_score);
            sum_exp = sum_exp + $rtoi(weight_r[t] * (1 << FRAC_W));
          end
          for (t = 0; t <= pos; t = t + 1) begin
            weights[t] = real_to_q(weight_r[t] / (sum_exp / real'(1 << FRAC_W)));
          end
        end else begin
          for (t = 0; t <= pos; t = t + 1) begin
            weights[t] = exp_approx(scores[t]);
            sum_exp = sum_exp + $signed(weights[t]);
          end
          for (t = 0; t <= pos; t = t + 1) begin
            weights[t] = div_q(weights[t], sum_exp[DATA_W-1:0]);
          end
        end
        for (d = 0; d < HEAD_DIM; d = d + 1) begin
          acc = '0;
          for (t = 0; t <= pos; t = t + 1) begin
            if (use_fp16) begin
              acc = acc + $signed(fp16_to_q(fp16_mul(weights[t],
                                                   v_cache[layer][idx2d(t, kv_head * HEAD_DIM + d, D_KV)])));
            end else begin
              acc = acc + ((weights[t] *
                            v_cache[layer][idx2d(t, kv_head * HEAD_DIM + d, D_KV)]) >>> FRAC_W);
            end
          end
          out[h * HEAD_DIM + d] = use_fp16 ? sat_q(acc) : sat_q(acc);
        end
      end
    end
  endtask

  task automatic compute_logits(
      input  logic signed [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic [31:0] token_id
  );
    integer tok;
    integer d;
    logic signed [ACC_W-1:0] acc;
    logic signed [ACC_W-1:0] best_acc;
    logic signed [ACC_W-1:0] top_score [0:4];
    logic [31:0] top_id [0:4];
    integer k_idx;
    integer insert;
    begin
      best_acc = -32'sh4000_0000;
      token_id = 0;
      for (k_idx = 0; k_idx < 5; k_idx = k_idx + 1) begin
        top_score[k_idx] = -32'sh4000_0000;
        top_id[k_idx] = 0;
      end
      for (tok = 0; tok < VOCAB; tok = tok + 1) begin
        acc = '0;
        for (d = 0; d < D_MODEL; d = d + 1) begin
          acc = acc + ((vec[d] * out_weight[idx2d(tok, d, D_MODEL)]) >>> FRAC_W);
        end
        if (acc > best_acc) begin
          best_acc = acc;
          token_id = tok[31:0];
        end
        if (acc > top_score[4]) begin
          insert = 4;
          while (insert > 0 && acc > top_score[insert-1]) begin
            top_score[insert] = top_score[insert-1];
            top_id[insert] = top_id[insert-1];
            insert = insert - 1;
          end
          top_score[insert] = acc;
          top_id[insert] = tok[31:0];
        end
      end
      for (k_idx = 0; k_idx < 5; k_idx = k_idx + 1) begin
        debug_topk_id[k_idx] = top_id[k_idx];
        debug_topk_score[k_idx] = top_score[k_idx];
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
            dump_this = dump_enable && (pos == dump_pos) &&
                        ((dump_all != 0) || (layer == dump_layer_arg));

            if (dump_this)
              dump_vec("x_in", layer, D_MODEL, x);
            rms_norm(layer, x, x_norm, attn_norm_w);
            if (dump_this)
              dump_vec("attn_norm", layer, D_MODEL, x_norm);
            matvec_q(layer, x_norm, q);
            matvec_k(layer, x_norm, k);
            matvec_v(layer, x_norm, v);
            apply_rope(pos, q, k);
            if (dump_this) begin
              dump_vec("q", layer, D_MODEL, q);
              dump_vec("k", layer, D_KV, k);
              dump_vec("v", layer, D_KV, v);
            end
            for (i = 0; i < D_KV; i = i + 1) begin
              k_cache[layer][idx2d(pos, i, D_KV)] = k[i];
              v_cache[layer][idx2d(pos, i, D_KV)] = v[i];
            end
            attention(layer, pos, q, attn_out);
            if (dump_this)
              dump_vec("attn_out", layer, D_MODEL, attn_out);
            matvec_attn_out(layer, attn_out, attn_proj);
            if (dump_this)
              dump_vec("attn_proj", layer, D_MODEL, attn_proj);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              if (use_fp16)
                x[i] = fp16_add(x[i], attn_proj[i]);
              else
                x[i] = add_q(x[i], attn_proj[i]);
            end
            if (dump_this)
              dump_vec("x_attn", layer, D_MODEL, x);

            rms_norm(layer, x, x_norm, ffn_norm_w);
            if (dump_this)
              dump_vec("ffn_norm", layer, D_MODEL, x_norm);
            matvec_ffn_gate(layer, x_norm, ffn_gate);
            matvec_ffn_up(layer, x_norm, ffn_up);
            if (dump_this) begin
              dump_vec("ffn_gate", layer, D_FF, ffn_gate);
              dump_vec("ffn_up", layer, D_FF, ffn_up);
            end
            for (i = 0; i < D_FF; i = i + 1) begin
              if (use_fp16)
                ffn_act[i] = fp16_mul(ffn_gate[i], ffn_up[i]);
              else
                ffn_act[i] = mul_q(silu_approx(ffn_gate[i]), ffn_up[i]);
            end
            if (dump_this)
              dump_vec("ffn_act", layer, D_FF, ffn_act);
            matvec_ffn_dn(layer, ffn_act, ffn_out);
            if (dump_this)
              dump_vec("ffn_out", layer, D_MODEL, ffn_out);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              if (use_fp16)
                x[i] = fp16_add(x[i], ffn_out[i]);
              else
                x[i] = add_q(x[i], ffn_out[i]);
            end
            if (dump_this)
              dump_vec("x_out", layer, D_MODEL, x);
          end
        end

        rms_norm_out(x, x_norm);
        compute_logits(x_norm, next_token_id);
        done <= 1'b1;
      end
    end
  end
  /* verilator lint_on BLKSEQ */
endmodule
