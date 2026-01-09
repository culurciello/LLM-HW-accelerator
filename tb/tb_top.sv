`timescale 1ns/1ps

`include "../src/transformer_accel.sv"

module tb_top;
  localparam int DATA_W = 16;
  localparam int FRAC_W = 8;

  logic clk;
  logic rst_n;
  logic host_we;
  logic host_re;
  logic [15:0] host_addr;
  logic signed [DATA_W-1:0] host_wdata;
  logic signed [DATA_W-1:0] host_rdata;
  logic start;
  logic done;

  transformer_accel dut (
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

  always #5 clk = ~clk;

  task automatic host_write(input [15:0] addr, input signed [DATA_W-1:0] data);
    begin
      @(negedge clk);
      host_we <= 1'b1;
      host_addr <= addr;
      host_wdata <= data;
      @(negedge clk);
      host_we <= 1'b0;
    end
  endtask

  task automatic host_read(input [15:0] addr, output signed [DATA_W-1:0] data);
    begin
      @(negedge clk);
      host_re <= 1'b1;
      host_addr <= addr;
      @(negedge clk);
      data = host_rdata;
      host_re <= 1'b0;
    end
  endtask

  function automatic signed [DATA_W-1:0] q(input real val);
    q = $rtoi(val * (1 << FRAC_W));
  endfunction

  integer i;
  logic signed [DATA_W-1:0] out_val;
  logic signed [DATA_W-1:0] expected [0:7];
  string weights_dir;
  string mem_path;
  bit load_weights;

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    host_we = 1'b0;
    host_re = 1'b0;
    host_addr = '0;
    host_wdata = '0;
    start = 1'b0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    host_write(16'h07F0, 2);

    load_weights = $value$plusargs("weights_dir=%s", weights_dir);
    if (load_weights) begin
      mem_path = {weights_dir, "/wq.mem"};
      $readmemh(mem_path, dut.wq_mem);
      mem_path = {weights_dir, "/wk.mem"};
      $readmemh(mem_path, dut.wk_mem);
      mem_path = {weights_dir, "/wv.mem"};
      $readmemh(mem_path, dut.wv_mem);
      mem_path = {weights_dir, "/wo.mem"};
      $readmemh(mem_path, dut.wo_mem);
      mem_path = {weights_dir, "/w1.mem"};
      $readmemh(mem_path, dut.w1_mem);
      mem_path = {weights_dir, "/w2.mem"};
      $readmemh(mem_path, dut.w2_mem);
    end

    if (!load_weights) begin
      for (i = 0; i < 256; i = i + 1) begin
        host_write(16'h0100 + i[15:0], 0);
        host_write(16'h0200 + i[15:0], 0);
        host_write(16'h0300 + i[15:0], 0);
        host_write(16'h0400 + i[15:0], 0);
        host_write(16'h0500 + i[15:0], 0);
        host_write(16'h0600 + i[15:0], 0);
      end
    end

    host_write(16'h0000, q(1.0));
    host_write(16'h0001, q(2.0));
    host_write(16'h0002, q(3.0));
    host_write(16'h0003, q(4.0));

    host_write(16'h0004, q(-1.0));
    host_write(16'h0005, q(0.5));
    host_write(16'h0006, q(-0.5));
    host_write(16'h0007, q(1.5));

    expected[0] = q(1.0);
    expected[1] = q(2.0);
    expected[2] = q(3.0);
    expected[3] = q(4.0);
    expected[4] = q(-1.0);
    expected[5] = q(0.5);
    expected[6] = q(-0.5);
    expected[7] = q(1.5);

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;

    i = 0;
    while (!done && i < 2000) begin
      @(negedge clk);
      i = i + 1;
    end

    if (!done) begin
      $display("Timeout waiting for done");
      $finish;
    end

    if (load_weights) begin
      for (i = 0; i < 8; i = i + 1) begin
        host_read(16'h0700 + i[15:0], out_val);
        $display("OUT[%0d]=%0d", i, out_val);
      end
      $display("PASS (weights loaded)");
    end else begin
      for (i = 0; i < 8; i = i + 1) begin
        host_read(16'h0700 + i[15:0], out_val);
        if (out_val !== expected[i]) begin
          $display("Mismatch at %0d: got %0d expected %0d", i, out_val, expected[i]);
          $finish;
        end
      end
      $display("PASS");
    end
    $finish;
  end
endmodule
