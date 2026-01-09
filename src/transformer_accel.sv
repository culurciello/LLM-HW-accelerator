`include "transformer_pkg.sv"
`include "matmul.sv"
`include "softmax.sv"
`include "layernorm.sv"

module transformer_accel #(
    parameter int DATA_W = transformer_pkg::DATA_W,
    parameter int FRAC_W = transformer_pkg::FRAC_W,
    parameter int ACC_W  = transformer_pkg::ACC_W,
    parameter int MAX_SEQ = 2,
    parameter int D_MODEL = 4,
    parameter int D_FF = 8,
    parameter int MATMUL_LANES = 1,
    parameter bit USE_FP16 = 0
  )(
    input  logic clk,
    input  logic rst_n,
    input  logic host_we,
    input  logic host_re,
    input  logic [15:0] host_addr,
    input  logic signed [DATA_W-1:0] host_wdata,
    output logic signed [DATA_W-1:0] host_rdata,
    input  logic start,
    output logic done
  );

  import transformer_pkg::*;

  localparam int INPUT_DEPTH   = MAX_SEQ * D_MODEL;
  localparam int WQ_DEPTH      = D_MODEL * D_MODEL;
  localparam int WK_DEPTH      = D_MODEL * D_MODEL;
  localparam int WV_DEPTH      = D_MODEL * D_MODEL;
  localparam int WO_DEPTH      = D_MODEL * D_MODEL;
  localparam int W1_DEPTH      = D_MODEL * D_FF;
  localparam int W2_DEPTH      = D_FF * D_MODEL;
  localparam int Q_DEPTH       = MAX_SEQ * D_MODEL;
  localparam int K_DEPTH       = MAX_SEQ * D_MODEL;
  localparam int V_DEPTH       = MAX_SEQ * D_MODEL;
  localparam int SCORE_DEPTH   = MAX_SEQ * MAX_SEQ;
  localparam int PROB_DEPTH    = MAX_SEQ * MAX_SEQ;
  localparam int CONTEXT_DEPTH = MAX_SEQ * D_MODEL;
  localparam int PROJ_DEPTH    = MAX_SEQ * D_MODEL;
  localparam int RES1_DEPTH    = MAX_SEQ * D_MODEL;
  localparam int HIDDEN_DEPTH  = MAX_SEQ * D_FF;
  localparam int ACT_DEPTH     = MAX_SEQ * D_FF;
  localparam int OUT_DEPTH     = MAX_SEQ * D_MODEL;

  localparam int ADDR_INPUT  = 16'h0000;
  localparam int ADDR_WQ     = 16'h0100;
  localparam int ADDR_WK     = 16'h0200;
  localparam int ADDR_WV     = 16'h0300;
  localparam int ADDR_WO     = 16'h0400;
  localparam int ADDR_W1     = 16'h0500;
  localparam int ADDR_W2     = 16'h0600;
  localparam int ADDR_OUTPUT = 16'h0700;
  localparam int ADDR_CTRL   = 16'h07F0;

  logic signed [DATA_W-1:0] input_mem  [0:INPUT_DEPTH-1];
  logic signed [DATA_W-1:0] wq_mem     [0:WQ_DEPTH-1];
  logic signed [DATA_W-1:0] wk_mem     [0:WK_DEPTH-1];
  logic signed [DATA_W-1:0] wv_mem     [0:WV_DEPTH-1];
  logic signed [DATA_W-1:0] wo_mem     [0:WO_DEPTH-1];
  logic signed [DATA_W-1:0] w1_mem     [0:W1_DEPTH-1];
  logic signed [DATA_W-1:0] w2_mem     [0:W2_DEPTH-1];

  logic signed [DATA_W-1:0] q_mem      [0:Q_DEPTH-1];
  logic signed [DATA_W-1:0] k_mem      [0:K_DEPTH-1];
  logic signed [DATA_W-1:0] v_mem      [0:V_DEPTH-1];
  logic signed [DATA_W-1:0] score_mem  [0:SCORE_DEPTH-1];
  logic signed [DATA_W-1:0] prob_mem   [0:PROB_DEPTH-1];
  logic signed [DATA_W-1:0] context_mem[0:CONTEXT_DEPTH-1];
  logic signed [DATA_W-1:0] proj_mem   [0:PROJ_DEPTH-1];
  logic signed [DATA_W-1:0] res1_mem   [0:RES1_DEPTH-1];
  logic signed [DATA_W-1:0] hidden_mem [0:HIDDEN_DEPTH-1];
  logic signed [DATA_W-1:0] act_mem    [0:ACT_DEPTH-1];
  logic signed [DATA_W-1:0] out_mem    [0:OUT_DEPTH-1];

  int unsigned seq_len;

  logic mm_q_start;
  logic mm_k_start;
  logic mm_v_start;
  logic mm_scores_start;
  logic sm_start;
  logic mm_context_start;
  logic mm_proj_start;
  logic mm_mlp1_start;
  logic mm_mlp2_start;

  logic mm_q_done;
  logic mm_k_done;
  logic mm_v_done;
  logic mm_scores_done;
  logic sm_done;
  logic mm_context_done;
  logic mm_proj_done;
  logic mm_mlp1_done;
  logic mm_mlp2_done;

  matmul #(
      .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
      .LANES(MATMUL_LANES),
      .USE_FP16(USE_FP16),
      .A_DEPTH(INPUT_DEPTH), .B_DEPTH(WQ_DEPTH), .C_DEPTH(Q_DEPTH)
    ) mm_q (
      .clk(clk), .rst_n(rst_n), .start(mm_q_start), .b_transpose(1'b0),
      .M(seq_len), .K(D_MODEL), .N(D_MODEL),
      .a_base(0), .b_base(0), .c_base(0),
      .done(mm_q_done),
      .a_mem(input_mem), .b_mem(wq_mem), .c_mem(q_mem)
    );

  matmul #(
      .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
      .LANES(MATMUL_LANES),
      .USE_FP16(USE_FP16),
      .A_DEPTH(INPUT_DEPTH), .B_DEPTH(WK_DEPTH), .C_DEPTH(K_DEPTH)
    ) mm_k (
      .clk(clk), .rst_n(rst_n), .start(mm_k_start), .b_transpose(1'b0),
      .M(seq_len), .K(D_MODEL), .N(D_MODEL),
      .a_base(0), .b_base(0), .c_base(0),
      .done(mm_k_done),
      .a_mem(input_mem), .b_mem(wk_mem), .c_mem(k_mem)
    );

  matmul #(
      .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
      .LANES(MATMUL_LANES),
      .USE_FP16(USE_FP16),
      .A_DEPTH(INPUT_DEPTH), .B_DEPTH(WV_DEPTH), .C_DEPTH(V_DEPTH)
    ) mm_v (
      .clk(clk), .rst_n(rst_n), .start(mm_v_start), .b_transpose(1'b0),
      .M(seq_len), .K(D_MODEL), .N(D_MODEL),
      .a_base(0), .b_base(0), .c_base(0),
      .done(mm_v_done),
      .a_mem(input_mem), .b_mem(wv_mem), .c_mem(v_mem)
    );

  matmul #(
      .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
      .LANES(MATMUL_LANES),
      .USE_FP16(USE_FP16),
      .A_DEPTH(Q_DEPTH), .B_DEPTH(K_DEPTH), .C_DEPTH(SCORE_DEPTH)
    ) mm_scores (
      .clk(clk), .rst_n(rst_n), .start(mm_scores_start), .b_transpose(1'b1),
      .M(seq_len), .K(D_MODEL), .N(seq_len),
      .a_base(0), .b_base(0), .c_base(0),
      .done(mm_scores_done),
      .a_mem(q_mem), .b_mem(k_mem), .c_mem(score_mem)
    );

  softmax #(
      .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
      .IN_DEPTH(SCORE_DEPTH), .OUT_DEPTH(PROB_DEPTH)
    ) sm (
      .clk(clk), .rst_n(rst_n), .start(sm_start),
      .M(seq_len), .N(seq_len),
      .in_base(0), .out_base(0),
      .done(sm_done),
      .in_mem(score_mem), .out_mem(prob_mem)
    );

  matmul #(
      .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
      .LANES(MATMUL_LANES),
      .USE_FP16(USE_FP16),
      .A_DEPTH(PROB_DEPTH), .B_DEPTH(V_DEPTH), .C_DEPTH(CONTEXT_DEPTH)
    ) mm_context (
      .clk(clk), .rst_n(rst_n), .start(mm_context_start), .b_transpose(1'b0),
      .M(seq_len), .K(seq_len), .N(D_MODEL),
      .a_base(0), .b_base(0), .c_base(0),
      .done(mm_context_done),
      .a_mem(prob_mem), .b_mem(v_mem), .c_mem(context_mem)
    );

  matmul #(
      .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
      .LANES(MATMUL_LANES),
      .USE_FP16(USE_FP16),
      .A_DEPTH(CONTEXT_DEPTH), .B_DEPTH(WO_DEPTH), .C_DEPTH(PROJ_DEPTH)
    ) mm_proj (
      .clk(clk), .rst_n(rst_n), .start(mm_proj_start), .b_transpose(1'b0),
      .M(seq_len), .K(D_MODEL), .N(D_MODEL),
      .a_base(0), .b_base(0), .c_base(0),
      .done(mm_proj_done),
      .a_mem(context_mem), .b_mem(wo_mem), .c_mem(proj_mem)
    );

  matmul #(
      .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
      .LANES(MATMUL_LANES),
      .USE_FP16(USE_FP16),
      .A_DEPTH(RES1_DEPTH), .B_DEPTH(W1_DEPTH), .C_DEPTH(HIDDEN_DEPTH)
    ) mm_mlp1 (
      .clk(clk), .rst_n(rst_n), .start(mm_mlp1_start), .b_transpose(1'b0),
      .M(seq_len), .K(D_MODEL), .N(D_FF),
      .a_base(0), .b_base(0), .c_base(0),
      .done(mm_mlp1_done),
      .a_mem(res1_mem), .b_mem(w1_mem), .c_mem(hidden_mem)
    );

  matmul #(
      .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
      .LANES(MATMUL_LANES),
      .USE_FP16(USE_FP16),
      .A_DEPTH(ACT_DEPTH), .B_DEPTH(W2_DEPTH), .C_DEPTH(OUT_DEPTH)
    ) mm_mlp2 (
      .clk(clk), .rst_n(rst_n), .start(mm_mlp2_start), .b_transpose(1'b0),
      .M(seq_len), .K(D_FF), .N(D_MODEL),
      .a_base(0), .b_base(0), .c_base(0),
      .done(mm_mlp2_done),
      .a_mem(act_mem), .b_mem(w2_mem), .c_mem(out_mem)
    );

  typedef enum logic [4:0] {
    S_IDLE,
    S_Q,
    S_K,
    S_V,
    S_SCORES,
    S_SOFTMAX,
    S_CONTEXT,
    S_PROJ,
    S_RES1,
    S_MLP1,
    S_ACT,
    S_MLP2,
    S_RES2,
    S_DONE
  } state_t;

  state_t state;

  int unsigned idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      done <= 1'b0;
      mm_q_start <= 1'b0;
      mm_k_start <= 1'b0;
      mm_v_start <= 1'b0;
      mm_scores_start <= 1'b0;
      sm_start <= 1'b0;
      mm_context_start <= 1'b0;
      mm_proj_start <= 1'b0;
      mm_mlp1_start <= 1'b0;
      mm_mlp2_start <= 1'b0;
      idx <= 0;
    end else begin
      case (state)
        S_IDLE: begin
          done <= 1'b0;
          if (start) begin
            mm_q_start <= 1'b1;
            state <= S_Q;
          end
        end
        S_Q: begin
          if (mm_q_done) begin
            mm_q_start <= 1'b0;
            mm_k_start <= 1'b1;
            state <= S_K;
          end
        end
        S_K: begin
          if (mm_k_done) begin
            mm_k_start <= 1'b0;
            mm_v_start <= 1'b1;
            state <= S_V;
          end
        end
        S_V: begin
          if (mm_v_done) begin
            mm_v_start <= 1'b0;
            mm_scores_start <= 1'b1;
            state <= S_SCORES;
          end
        end
        S_SCORES: begin
          if (mm_scores_done) begin
            mm_scores_start <= 1'b0;
            sm_start <= 1'b1;
            state <= S_SOFTMAX;
          end
        end
        S_SOFTMAX: begin
          if (sm_done) begin
            sm_start <= 1'b0;
            mm_context_start <= 1'b1;
            state <= S_CONTEXT;
          end
        end
        S_CONTEXT: begin
          if (mm_context_done) begin
            mm_context_start <= 1'b0;
            mm_proj_start <= 1'b1;
            state <= S_PROJ;
          end
        end
        S_PROJ: begin
          if (mm_proj_done) begin
            mm_proj_start <= 1'b0;
            idx <= 0;
            state <= S_RES1;
          end
        end
        S_RES1: begin
          res1_mem[idx] <= add_q(input_mem[idx], proj_mem[idx]);
          if (idx == INPUT_DEPTH - 1) begin
            idx <= 0;
            mm_mlp1_start <= 1'b1;
            state <= S_MLP1;
          end else begin
            idx <= idx + 1;
          end
        end
        S_MLP1: begin
          if (mm_mlp1_done) begin
            mm_mlp1_start <= 1'b0;
            idx <= 0;
            state <= S_ACT;
          end
        end
        S_ACT: begin
          act_mem[idx] <= gelu_approx(hidden_mem[idx]);
          if (idx == HIDDEN_DEPTH - 1) begin
            idx <= 0;
            mm_mlp2_start <= 1'b1;
            state <= S_MLP2;
          end else begin
            idx <= idx + 1;
          end
        end
        S_MLP2: begin
          if (mm_mlp2_done) begin
            mm_mlp2_start <= 1'b0;
            idx <= 0;
            state <= S_RES2;
          end
        end
        S_RES2: begin
          out_mem[idx] <= add_q(res1_mem[idx], out_mem[idx]);
          if (idx == OUT_DEPTH - 1) begin
            state <= S_DONE;
          end else begin
            idx <= idx + 1;
          end
        end
        S_DONE: begin
          done <= 1'b1;
          if (!start)
            state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      seq_len <= 1;
    end else if (host_we && host_addr == ADDR_CTRL) begin
      seq_len <= host_wdata[7:0];
    end
  end

  always_comb begin
    host_rdata = '0;
    if (host_re) begin
      if (host_addr >= ADDR_INPUT && host_addr < ADDR_INPUT + INPUT_DEPTH)
        host_rdata = input_mem[host_addr - ADDR_INPUT];
      else if (host_addr >= ADDR_WQ && host_addr < ADDR_WQ + WQ_DEPTH)
        host_rdata = wq_mem[host_addr - ADDR_WQ];
      else if (host_addr >= ADDR_WK && host_addr < ADDR_WK + WK_DEPTH)
        host_rdata = wk_mem[host_addr - ADDR_WK];
      else if (host_addr >= ADDR_WV && host_addr < ADDR_WV + WV_DEPTH)
        host_rdata = wv_mem[host_addr - ADDR_WV];
      else if (host_addr >= ADDR_WO && host_addr < ADDR_WO + WO_DEPTH)
        host_rdata = wo_mem[host_addr - ADDR_WO];
      else if (host_addr >= ADDR_W1 && host_addr < ADDR_W1 + W1_DEPTH)
        host_rdata = w1_mem[host_addr - ADDR_W1];
      else if (host_addr >= ADDR_W2 && host_addr < ADDR_W2 + W2_DEPTH)
        host_rdata = w2_mem[host_addr - ADDR_W2];
      else if (host_addr >= ADDR_OUTPUT && host_addr < ADDR_OUTPUT + OUT_DEPTH)
        host_rdata = out_mem[host_addr - ADDR_OUTPUT];
      else if (host_addr == ADDR_CTRL)
        host_rdata = seq_len[DATA_W-1:0];
      else if (host_addr == ADDR_CTRL + 1)
        host_rdata = {{(DATA_W-1){1'b0}}, done};
    end
  end

  always_ff @(posedge clk) begin
    if (host_we) begin
      if (host_addr >= ADDR_INPUT && host_addr < ADDR_INPUT + INPUT_DEPTH)
        input_mem[host_addr - ADDR_INPUT] <= host_wdata;
      else if (host_addr >= ADDR_WQ && host_addr < ADDR_WQ + WQ_DEPTH)
        wq_mem[host_addr - ADDR_WQ] <= host_wdata;
      else if (host_addr >= ADDR_WK && host_addr < ADDR_WK + WK_DEPTH)
        wk_mem[host_addr - ADDR_WK] <= host_wdata;
      else if (host_addr >= ADDR_WV && host_addr < ADDR_WV + WV_DEPTH)
        wv_mem[host_addr - ADDR_WV] <= host_wdata;
      else if (host_addr >= ADDR_WO && host_addr < ADDR_WO + WO_DEPTH)
        wo_mem[host_addr - ADDR_WO] <= host_wdata;
      else if (host_addr >= ADDR_W1 && host_addr < ADDR_W1 + W1_DEPTH)
        w1_mem[host_addr - ADDR_W1] <= host_wdata;
      else if (host_addr >= ADDR_W2 && host_addr < ADDR_W2 + W2_DEPTH)
        w2_mem[host_addr - ADDR_W2] <= host_wdata;
    end
  end
endmodule
