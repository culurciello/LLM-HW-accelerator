`include "transformer_pkg.sv"

module layernorm #(
    parameter int DATA_W = transformer_pkg::DATA_W,
    parameter int FRAC_W = transformer_pkg::FRAC_W,
    parameter int ACC_W  = transformer_pkg::ACC_W,
    parameter int IN_DEPTH = 256,
    parameter int OUT_DEPTH = 256
  )(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  int unsigned M,
    input  int unsigned N,
    input  int unsigned in_base,
    input  int unsigned out_base,
    output logic done,
    input  logic signed [DATA_W-1:0] in_mem [0:IN_DEPTH-1],
    output logic signed [DATA_W-1:0] out_mem [0:OUT_DEPTH-1]
  );

  import transformer_pkg::*;

  typedef enum logic [1:0] {IDLE, MEAN, APPLY, DONE} state_t;
  state_t state;

  int unsigned row;
  int unsigned col;
  logic signed [ACC_W-1:0] sum;
  logic signed [DATA_W-1:0] mean;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      done <= 1'b0;
      row <= 0;
      col <= 0;
      sum <= '0;
      mean <= '0;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          if (start) begin
            row <= 0;
            col <= 0;
            sum <= '0;
            state <= MEAN;
          end
        end
        MEAN: begin
          logic signed [DATA_W-1:0] val;
          logic signed [ACC_W-1:0] next_sum;
          val = in_mem[in_base + (row * N) + col];
          next_sum = sum + val;
          sum <= next_sum;
          if (col == N - 1) begin
            mean <= next_sum[DATA_W-1:0] / N;
            col <= 0;
            state <= APPLY;
          end else begin
            col <= col + 1;
          end
        end
        APPLY: begin
          logic signed [DATA_W-1:0] val;
          val = in_mem[in_base + (row * N) + col];
          out_mem[out_base + (row * N) + col] <= val - mean;

          if (col == N - 1) begin
            col <= 0;
            sum <= '0;
            if (row == M - 1) begin
              done <= 1'b1;
              if (!start)
                state <= IDLE;
            end else begin
              row <= row + 1;
              state <= MEAN;
            end
          end else begin
            col <= col + 1;
          end
        end
        default: state <= IDLE;
      endcase
    end
  end
endmodule
