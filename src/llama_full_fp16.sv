`timescale 1ns/1ps
`include "llama_full.sv"

module llama_full_fp16 #(
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

  llama_full #(
      .DATA_W(DATA_W),
      .FRAC_W(FRAC_W),
      .ACC_W(ACC_W),
      .N_LAYER(N_LAYER),
      .N_HEAD(N_HEAD),
      .N_KV_HEAD(N_KV_HEAD),
      .D_MODEL(D_MODEL),
      .D_FF(D_FF),
      .MAX_CTX(MAX_CTX),
      .VOCAB(VOCAB),
      .ROPE_DIM(ROPE_DIM),
      .ROPE_BASE(ROPE_BASE),
      .RMS_EPS(RMS_EPS)
    ) dut (
      .clk(clk),
      .rst_n(rst_n),
      .start(start),
      .prompt_len(prompt_len),
      .done(done),
      .next_token_id(next_token_id)
    );
endmodule
