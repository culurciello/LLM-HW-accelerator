`include "transformer_pkg.sv"

module transformer_full #(
    parameter int DATA_W = transformer_pkg::DATA_W,
    parameter int FRAC_W = transformer_pkg::FRAC_W,
    parameter int ACC_W  = transformer_pkg::ACC_W,
    parameter int N_LAYER = 8,
    parameter int N_HEAD  = 16,
    parameter int D_MODEL = 64,
    parameter int D_FF    = 256,
    parameter int MAX_CTX = 2048,
    parameter int VOCAB   = 50257,
    parameter int MAX_PROMPT = 128,
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

  logic signed [DATA_W-1:0] token_embd [0:VOCAB*D_MODEL-1];
  logic signed [DATA_W-1:0] pos_embd   [0:MAX_CTX*D_MODEL-1];
  logic signed [DATA_W-1:0] out_weight [0:VOCAB*D_MODEL-1];
  logic signed [DATA_W-1:0] out_norm_w [0:D_MODEL-1];
  logic signed [DATA_W-1:0] out_norm_b [0:D_MODEL-1];

  logic signed [DATA_W-1:0] attn_norm_w [0:N_LAYER*D_MODEL-1];
  logic signed [DATA_W-1:0] attn_norm_b [0:N_LAYER*D_MODEL-1];
  logic signed [DATA_W-1:0] qkv_w       [0:N_LAYER*D_MODEL*3*D_MODEL-1];
  logic signed [DATA_W-1:0] qkv_b       [0:N_LAYER*3*D_MODEL-1];
  logic signed [DATA_W-1:0] attn_out_w  [0:N_LAYER*D_MODEL*D_MODEL-1];
  logic signed [DATA_W-1:0] attn_out_b  [0:N_LAYER*D_MODEL-1];
  logic signed [DATA_W-1:0] ffn_norm_w  [0:N_LAYER*D_MODEL-1];
  logic signed [DATA_W-1:0] ffn_norm_b  [0:N_LAYER*D_MODEL-1];
  logic signed [DATA_W-1:0] ffn_up_w    [0:N_LAYER*D_MODEL*D_FF-1];
  logic signed [DATA_W-1:0] ffn_up_b    [0:N_LAYER*D_FF-1];
  logic signed [DATA_W-1:0] ffn_dn_w    [0:N_LAYER*D_FF*D_MODEL-1];
  logic signed [DATA_W-1:0] ffn_dn_b    [0:N_LAYER*D_MODEL-1];

  logic [31:0] input_tokens [0:MAX_PROMPT-1];

  logic signed [DATA_W-1:0] k_cache [0:N_LAYER-1][0:MAX_CTX*D_MODEL-1];
  logic signed [DATA_W-1:0] v_cache [0:N_LAYER-1][0:MAX_CTX*D_MODEL-1];

  logic signed [DATA_W-1:0] x        [0:D_MODEL-1];
  logic signed [DATA_W-1:0] x_norm   [0:D_MODEL-1];
  logic signed [DATA_W-1:0] qkv      [0:3*D_MODEL-1];
  logic signed [DATA_W-1:0] attn_out [0:D_MODEL-1];
  logic signed [DATA_W-1:0] attn_proj[0:D_MODEL-1];
  logic signed [DATA_W-1:0] ffn_hid  [0:D_FF-1];
  logic signed [DATA_W-1:0] ffn_act  [0:D_FF-1];
  logic signed [DATA_W-1:0] ffn_out  [0:D_MODEL-1];
  logic signed [DATA_W-1:0] scores   [0:MAX_CTX-1];
  logic signed [DATA_W-1:0] weights  [0:MAX_CTX-1];
  logic [31:0] debug_topk_id [0:TOP_K-1];
  logic signed [ACC_W-1:0] debug_topk_score [0:TOP_K-1];

  function automatic int unsigned idx2d(
      input int unsigned row,
      input int unsigned col,
      input int unsigned stride
  );
    return row * stride + col;
  endfunction

  function automatic logic signed [DATA_W-1:0] ln_apply(
      input logic signed [DATA_W-1:0] val,
      input logic signed [DATA_W-1:0] mean_q,
      input logic signed [DATA_W-1:0] inv_std_q,
      input logic signed [DATA_W-1:0] w_q,
      input logic signed [DATA_W-1:0] b_q
  );
    logic signed [DATA_W-1:0] normed;
    begin
      normed = mul_q(val - mean_q, inv_std_q);
      ln_apply = add_q(mul_q(normed, w_q), b_q);
    end
  endfunction

  task automatic layer_norm_attn(
      input  int unsigned layer,
      input  logic signed [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out_vec[0:D_MODEL-1]
  );
    integer i;
    logic signed [ACC_W-1:0] sum;
    logic signed [ACC_W-1:0] mean_acc;
    logic signed [ACC_W-1:0] var_acc;
    logic signed [DATA_W-1:0] mean_q;
    logic signed [DATA_W-1:0] inv_std_q;
    real var_r;
    real inv_std_r;
    begin
      sum = '0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        sum = sum + in_vec[i];
      end
      mean_acc = sum / D_MODEL;
      mean_q = sat_q(mean_acc);
      var_acc = '0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        logic signed [ACC_W-1:0] diff;
        diff = in_vec[i] - mean_q;
        var_acc = var_acc + ((diff * diff) >>> FRAC_W);
      end
      var_acc = var_acc / D_MODEL;
      var_r = var_acc / real'(1 << FRAC_W);
      inv_std_r = 1.0 / $sqrt(var_r + 1.0e-5);
      inv_std_q = $rtoi(inv_std_r * (1 << FRAC_W));
      for (i = 0; i < D_MODEL; i = i + 1) begin
        out_vec[i] = ln_apply(
          in_vec[i],
          mean_q,
          inv_std_q,
          attn_norm_w[layer * D_MODEL + i],
          attn_norm_b[layer * D_MODEL + i]
        );
      end
    end
  endtask

  task automatic layer_norm_ffn(
      input  int unsigned layer,
      input  logic signed [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out_vec[0:D_MODEL-1]
  );
    integer i;
    logic signed [ACC_W-1:0] sum;
    logic signed [ACC_W-1:0] mean_acc;
    logic signed [ACC_W-1:0] var_acc;
    logic signed [DATA_W-1:0] mean_q;
    logic signed [DATA_W-1:0] inv_std_q;
    real var_r;
    real inv_std_r;
    begin
      sum = '0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        sum = sum + in_vec[i];
      end
      mean_acc = sum / D_MODEL;
      mean_q = sat_q(mean_acc);
      var_acc = '0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        logic signed [ACC_W-1:0] diff;
        diff = in_vec[i] - mean_q;
        var_acc = var_acc + ((diff * diff) >>> FRAC_W);
      end
      var_acc = var_acc / D_MODEL;
      var_r = var_acc / real'(1 << FRAC_W);
      inv_std_r = 1.0 / $sqrt(var_r + 1.0e-5);
      inv_std_q = $rtoi(inv_std_r * (1 << FRAC_W));
      for (i = 0; i < D_MODEL; i = i + 1) begin
        out_vec[i] = ln_apply(
          in_vec[i],
          mean_q,
          inv_std_q,
          ffn_norm_w[layer * D_MODEL + i],
          ffn_norm_b[layer * D_MODEL + i]
        );
      end
    end
  endtask

  task automatic layer_norm_out(
      input  logic signed [DATA_W-1:0] in_vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out_vec[0:D_MODEL-1]
  );
    integer i;
    logic signed [ACC_W-1:0] sum;
    logic signed [ACC_W-1:0] mean_acc;
    logic signed [ACC_W-1:0] var_acc;
    logic signed [DATA_W-1:0] mean_q;
    logic signed [DATA_W-1:0] inv_std_q;
    real var_r;
    real inv_std_r;
    begin
      sum = '0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        sum = sum + in_vec[i];
      end
      mean_acc = sum / D_MODEL;
      mean_q = sat_q(mean_acc);
      var_acc = '0;
      for (i = 0; i < D_MODEL; i = i + 1) begin
        logic signed [ACC_W-1:0] diff;
        diff = in_vec[i] - mean_q;
        var_acc = var_acc + ((diff * diff) >>> FRAC_W);
      end
      var_acc = var_acc / D_MODEL;
      var_r = var_acc / real'(1 << FRAC_W);
      inv_std_r = 1.0 / $sqrt(var_r + 1.0e-5);
      inv_std_q = $rtoi(inv_std_r * (1 << FRAC_W));
      for (i = 0; i < D_MODEL; i = i + 1) begin
        out_vec[i] = ln_apply(
          in_vec[i],
          mean_q,
          inv_std_q,
          out_norm_w[i],
          out_norm_b[i]
        );
      end
    end
  endtask

  task automatic matvec_attn_out(
      input  int unsigned layer,
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
        out[j] = add_q(sat_q(acc), attn_out_b[layer*D_MODEL + j]);
      end
    end
  endtask

  task automatic matvec_qkv_layer(
      input  int unsigned layer,
      input  logic signed [DATA_W-1:0] vec [0:D_MODEL-1],
      output logic signed [DATA_W-1:0] out [0:3*D_MODEL-1]
  );
    integer i;
    integer j;
    logic signed [ACC_W-1:0] acc;
    begin
      for (j = 0; j < 3*D_MODEL; j = j + 1) begin
        acc = '0;
        for (i = 0; i < D_MODEL; i = i + 1) begin
          acc = acc + ((vec[i] * qkv_w[layer*D_MODEL*3*D_MODEL + i*3*D_MODEL + j]) >>> FRAC_W);
        end
        out[j] = add_q(sat_q(acc), qkv_b[layer*3*D_MODEL + j]);
      end
    end
  endtask

  task automatic matvec_ffn_up(
      input  int unsigned layer,
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
        out[j] = add_q(sat_q(acc), ffn_up_b[layer*D_FF + j]);
      end
    end
  endtask

  task automatic matvec_ffn_down(
      input  int unsigned layer,
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
        out[j] = add_q(sat_q(acc), ffn_dn_b[layer*D_MODEL + j]);
      end
    end
  endtask

  task automatic attention(
      input  int unsigned layer,
      input  int unsigned pos,
      input  logic signed [DATA_W-1:0] qkv_vec [0:3*D_MODEL-1],
      output logic signed [DATA_W-1:0] out [0:D_MODEL-1]
  );
    integer h;
    integer t;
    integer d;
    integer i;
    logic signed [DATA_W-1:0] qh;
    logic signed [DATA_W-1:0] kh;
    logic signed [DATA_W-1:0] vh;
    logic signed [ACC_W-1:0] acc;
    logic signed [DATA_W-1:0] max_score;
    logic signed [ACC_W-1:0] sum_exp;
    logic signed [DATA_W-1:0] scale_q;
    real scale_r;
    begin
      scale_r = 1.0 / $sqrt(HEAD_DIM);
      scale_q = $rtoi(scale_r * (1 << FRAC_W));
      for (d = 0; d < D_MODEL; d = d + 1)
        out[d] = '0;

      for (h = 0; h < N_HEAD; h = h + 1) begin
        for (t = 0; t <= pos; t = t + 1) begin
          acc = '0;
          for (d = 0; d < HEAD_DIM; d = d + 1) begin
            i = h * HEAD_DIM + d;
            qh = qkv_vec[i];
            kh = k_cache[layer][idx2d(t, i, D_MODEL)];
            acc = acc + ((qh * kh) >>> FRAC_W);
          end
          scores[t] = mul_q(sat_q(acc), scale_q);
        end
        max_score = scores[0];
        for (t = 1; t <= pos; t = t + 1) begin
          if (scores[t] > max_score)
            max_score = scores[t];
        end
        sum_exp = '0;
        for (t = 0; t <= pos; t = t + 1) begin
          weights[t] = exp_approx(scores[t] - max_score);
          sum_exp = sum_exp + weights[t];
        end
        for (t = 0; t <= pos; t = t + 1) begin
          weights[t] = div_q(weights[t], sum_exp[DATA_W-1:0]);
        end
        for (d = 0; d < HEAD_DIM; d = d + 1) begin
          acc = '0;
          for (t = 0; t <= pos; t = t + 1) begin
            i = h * HEAD_DIM + d;
            vh = v_cache[layer][idx2d(t, i, D_MODEL)];
            acc = acc + ((weights[t] * vh) >>> FRAC_W);
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
    logic signed [ACC_W-1:0] top_score [0:TOP_K-1];
    logic [31:0] top_id [0:TOP_K-1];
    integer k;
    integer insert;
    begin
      best_acc = -32'sh4000_0000;
      token_id = 0;
      for (k = 0; k < TOP_K; k = k + 1) begin
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
        if (acc > top_score[TOP_K-1]) begin
          insert = TOP_K-1;
          while (insert > 0 && acc > top_score[insert-1]) begin
            top_score[insert] = top_score[insert-1];
            top_id[insert] = top_id[insert-1];
            insert = insert - 1;
          end
          top_score[insert] = acc;
          top_id[insert] = v[31:0];
        end
      end
      for (k = 0; k < TOP_K; k = k + 1) begin
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
            for (i = 0; i < D_MODEL; i = i + 1) begin
              k_cache[layer][idx2d(pos, i, D_MODEL)] <= '0;
              v_cache[layer][idx2d(pos, i, D_MODEL)] <= '0;
            end
          end
        end

        for (pos = 0; pos < prompt_len; pos = pos + 1) begin
          for (i = 0; i < D_MODEL; i = i + 1) begin
            x[i] = add_q(
              token_embd[idx2d(input_tokens[pos], i, D_MODEL)],
              pos_embd[idx2d(pos, i, D_MODEL)]
            );
          end

          for (layer = 0; layer < N_LAYER; layer = layer + 1) begin
            layer_norm_attn(layer, x, x_norm);
            matvec_qkv_layer(layer, x_norm, qkv);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              k_cache[layer][idx2d(pos, i, D_MODEL)] = qkv[D_MODEL + i];
              v_cache[layer][idx2d(pos, i, D_MODEL)] = qkv[2*D_MODEL + i];
            end
            attention(layer, pos, qkv, attn_out);
            matvec_attn_out(layer, attn_out, attn_proj);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              x[i] = add_q(x[i], attn_proj[i]);
            end

            layer_norm_ffn(layer, x, x_norm);
            matvec_ffn_up(layer, x_norm, ffn_hid);
            for (i = 0; i < D_FF; i = i + 1)
              ffn_act[i] = gelu_approx(ffn_hid[i]);
            matvec_ffn_down(layer, ffn_act, ffn_out);
            for (i = 0; i < D_MODEL; i = i + 1) begin
              x[i] = add_q(x[i], ffn_out[i]);
            end
          end
        end

        layer_norm_out(x, x_norm);
        compute_logits(x_norm, next_token_id);
        done <= 1'b1;
      end
    end
  end
endmodule
