`include "transformer_pkg.sv"

module softmax #(
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

  typedef enum logic [1:0] {IDLE, MAX_SUM, WRITE, DONE} state_t;
  state_t state;

  int unsigned row;
  int unsigned col;
  logic signed [DATA_W-1:0] row_max;
  logic signed [ACC_W-1:0] row_sum;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      done <= 1'b0;
      row <= 0;
      col <= 0;
      row_max <= '0;
      row_sum <= '0;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          if (start) begin
            row <= 0;
            col <= 0;
            row_max <= in_mem[in_base];
            row_sum <= '0;
            state <= MAX_SUM;
          end
        end
        MAX_SUM: begin
          logic signed [DATA_W-1:0] val;
          val = in_mem[in_base + (row * N) + col];
          if (val > row_max)
            row_max <= val;

          if (col == N - 1) begin
            col <= 0;
            row_sum <= '0;
            state <= WRITE;
          end else begin
            col <= col + 1;
          end
        end
        WRITE: begin
          logic signed [DATA_W-1:0] val;
          logic signed [DATA_W-1:0] expv;
          val = in_mem[in_base + (row * N) + col];
          expv = exp_approx(val - row_max);
          row_sum <= row_sum + expv;

          if (col == N - 1) begin
            col <= 0;
            state <= DONE;
          end else begin
            col <= col + 1;
          end
        end
        DONE: begin
          logic signed [DATA_W-1:0] val;
          logic signed [DATA_W-1:0] expv;
          logic signed [DATA_W-1:0] norm;

          val = in_mem[in_base + (row * N) + col];
          expv = exp_approx(val - row_max);
          norm = div_q(expv, row_sum[DATA_W-1:0]);
          out_mem[out_base + (row * N) + col] <= norm;

          if (col == N - 1) begin
            col <= 0;
            if (row == M - 1) begin
              done <= 1'b1;
              if (!start)
                state <= IDLE;
            end else begin
              row <= row + 1;
              row_max <= in_mem[in_base + ((row + 1) * N)];
              row_sum <= '0;
              state <= MAX_SUM;
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
