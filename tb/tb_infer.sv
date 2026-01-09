`timescale 1ns/1ps

`include "../src/transformer_full.sv"

module tb_infer;
  localparam int DATA_W = 16;

  logic clk;
  logic rst_n;
  logic start;
  logic done;
  logic [31:0] next_token_id;
  int unsigned prompt_len;
  string weights_dir;
  string prompt_path;
  string mem_path;
  int unsigned prompt_len_arg;
  int unsigned dump_topk;

  transformer_full dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .prompt_len(prompt_len),
    .done(done),
    .next_token_id(next_token_id)
  );

  always #5 clk = ~clk;

  task automatic load_mem(input string path, output integer ok);
    begin
      ok = 1;
      if (path.len() == 0) begin
        ok = 0;
      end
    end
  endtask

  integer i;
  integer ok;

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start = 1'b0;
    prompt_len = 0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    if (!$value$plusargs("weights_dir=%s", weights_dir)) begin
      $display("Missing +weights_dir");
      $finish;
    end
    if (!$value$plusargs("prompt_ids=%s", prompt_path)) begin
      $display("Missing +prompt_ids");
      $finish;
    end

    mem_path = {weights_dir, "/token_embd.mem"};
    $readmemh(mem_path, dut.token_embd);
    mem_path = {weights_dir, "/pos_embd.mem"};
    $readmemh(mem_path, dut.pos_embd);
    mem_path = {weights_dir, "/output_weight.mem"};
    $readmemh(mem_path, dut.out_weight);
    mem_path = {weights_dir, "/output_norm_weight.mem"};
    $readmemh(mem_path, dut.out_norm_w);
    mem_path = {weights_dir, "/output_norm_bias.mem"};
    $readmemh(mem_path, dut.out_norm_b);

    mem_path = {weights_dir, "/attn_norm_weight.mem"};
    $readmemh(mem_path, dut.attn_norm_w);
    mem_path = {weights_dir, "/attn_norm_bias.mem"};
    $readmemh(mem_path, dut.attn_norm_b);
    mem_path = {weights_dir, "/attn_qkv_weight.mem"};
    $readmemh(mem_path, dut.qkv_w);
    mem_path = {weights_dir, "/attn_qkv_bias.mem"};
    $readmemh(mem_path, dut.qkv_b);
    mem_path = {weights_dir, "/attn_output_weight.mem"};
    $readmemh(mem_path, dut.attn_out_w);
    mem_path = {weights_dir, "/attn_output_bias.mem"};
    $readmemh(mem_path, dut.attn_out_b);
    mem_path = {weights_dir, "/ffn_norm_weight.mem"};
    $readmemh(mem_path, dut.ffn_norm_w);
    mem_path = {weights_dir, "/ffn_norm_bias.mem"};
    $readmemh(mem_path, dut.ffn_norm_b);
    mem_path = {weights_dir, "/ffn_up_weight.mem"};
    $readmemh(mem_path, dut.ffn_up_w);
    mem_path = {weights_dir, "/ffn_up_bias.mem"};
    $readmemh(mem_path, dut.ffn_up_b);
    mem_path = {weights_dir, "/ffn_down_weight.mem"};
    $readmemh(mem_path, dut.ffn_dn_w);
    mem_path = {weights_dir, "/ffn_down_bias.mem"};
    $readmemh(mem_path, dut.ffn_dn_b);

    $readmemh(prompt_path, dut.input_tokens);
    if (!$value$plusargs("prompt_len=%d", prompt_len_arg)) begin
      $display("Missing +prompt_len");
      $finish;
    end
    prompt_len = prompt_len_arg;

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    i = 0;
    while (!done && i < 200) begin
      @(negedge clk);
      i = i + 1;
    end

    if (!done) begin
      $display("Timeout waiting for done");
      $finish;
    end

    $display("NEXT_TOKEN_ID=%0d", next_token_id);
    if ($value$plusargs("dump_topk=%d", dump_topk) && dump_topk != 0) begin
      for (i = 0; i < 5; i = i + 1) begin
        $display("TOPK[%0d] id=%0d score=%0d", i, dut.debug_topk_id[i], dut.debug_topk_score[i]);
      end
    end
    $finish;
  end
endmodule
