`include "transformer_pkg.sv"
`include "board_presets.sv"
`include "transformer_accel.sv"

module transformer_accel_xcvu19p (
    input  logic clk,
    input  logic rst_n,
    input  logic host_we,
    input  logic host_re,
    input  logic [15:0] host_addr,
    input  logic signed [transformer_pkg::DATA_W-1:0] host_wdata,
    output logic signed [transformer_pkg::DATA_W-1:0] host_rdata,
    input  logic start,
    output logic done
  );

  import board_presets::*;

  transformer_accel #(
      .MAX_SEQ(XCVU19P_MAX_SEQ),
      .D_MODEL(XCVU19P_D_MODEL),
      .D_FF(XCVU19P_D_FF),
      .MATMUL_LANES(XCVU19P_MATMUL_LANES)
    ) u_accel (
      .clk(clk),
      .rst_n(rst_n),
      .host_we(host_we),
      .host_re(host_re),
      .host_addr(host_addr),
      .host_wdata(host_wdata),
      .host_rdata(host_rdata),
      .start(start),
      .done(done)
    );
endmodule
