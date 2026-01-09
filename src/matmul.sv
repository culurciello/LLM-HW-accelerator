`include "transformer_pkg.sv"

module matmul #(
    parameter int DATA_W = transformer_pkg::DATA_W,
    parameter int FRAC_W = transformer_pkg::FRAC_W,
    parameter int ACC_W  = transformer_pkg::ACC_W,
    parameter int LANES  = 1,
    parameter int A_DEPTH = 256,
    parameter int B_DEPTH = 256,
    parameter int C_DEPTH = 256
  )(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic b_transpose,
    input  int unsigned M,
    input  int unsigned K,
    input  int unsigned N,
    input  int unsigned a_base,
    input  int unsigned b_base,
    input  int unsigned c_base,
    output logic done,
    input  logic signed [DATA_W-1:0] a_mem [0:A_DEPTH-1],
    input  logic signed [DATA_W-1:0] b_mem [0:B_DEPTH-1],
    output logic signed [DATA_W-1:0] c_mem [0:C_DEPTH-1]
  );

  import transformer_pkg::*;

  typedef enum logic [1:0] {IDLE, RUN, DONE} state_t;
  state_t state;

  int unsigned i;
  int unsigned j;
  int unsigned k;
  logic signed [ACC_W-1:0] acc;
  logic signed [ACC_W-1:0] lane_sum;
  integer lane;

  function automatic int unsigned b_index(
      input int unsigned k_idx,
      input int unsigned j_idx
  );
    begin
      if (b_transpose)
        b_index = b_base + (j_idx * K) + k_idx;
      else
        b_index = b_base + (k_idx * N) + j_idx;
    end
  endfunction

  always_comb begin
    lane_sum = '0;
    for (lane = 0; lane < LANES; lane = lane + 1) begin
      if (k + lane < K) begin
        lane_sum = lane_sum + (a_mem[a_base + (i * K) + (k + lane)] *
                               b_mem[b_index(k + lane, j)] >>> FRAC_W);
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      done  <= 1'b0;
      i     <= 0;
      j     <= 0;
      k     <= 0;
      acc   <= '0;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          if (start) begin
            i   <= 0;
            j   <= 0;
            k   <= 0;
            acc <= '0;
            state <= RUN;
          end
        end
        RUN: begin
          logic signed [ACC_W-1:0] mac;

          mac   = acc + lane_sum;

          if (k + LANES >= K) begin
            c_mem[c_base + (i * N) + j] <= sat_q(mac);
            acc <= '0;
            k <= 0;
            if (j == N - 1) begin
              j <= 0;
              if (i == M - 1) begin
                state <= DONE;
              end else begin
                i <= i + 1;
              end
            end else begin
              j <= j + 1;
            end
          end else begin
            k <= k + 1;
            acc <= mac;
          end
        end
        DONE: begin
          done <= 1'b1;
          if (!start)
            state <= IDLE;
        end
        default: state <= IDLE;
      endcase
    end
  end
endmodule
