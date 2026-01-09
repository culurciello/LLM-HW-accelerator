`ifndef TRANSFORMER_PKG_SV
`define TRANSFORMER_PKG_SV

package transformer_pkg;
  parameter int DATA_W = 16;
  parameter int FRAC_W = 8;
  parameter int ACC_W  = 32;

  function automatic logic signed [DATA_W-1:0] sat_q(
      input logic signed [ACC_W-1:0] v
  );
    logic signed [DATA_W-1:0] max_v;
    logic signed [DATA_W-1:0] min_v;
    begin
      max_v = {1'b0, {(DATA_W-1){1'b1}}};
      min_v = {1'b1, {(DATA_W-1){1'b0}}};
      if (v > max_v)
        sat_q = max_v;
      else if (v < min_v)
        sat_q = min_v;
      else
        sat_q = v[DATA_W-1:0];
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] mul_q(
      input logic signed [DATA_W-1:0] a,
      input logic signed [DATA_W-1:0] b
  );
    logic signed [2*DATA_W-1:0] prod;
    logic signed [ACC_W-1:0] shifted;
    begin
      prod = a * b;
      shifted = prod >>> FRAC_W;
      mul_q = sat_q(shifted);
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] add_q(
      input logic signed [DATA_W-1:0] a,
      input logic signed [DATA_W-1:0] b
  );
    logic signed [ACC_W-1:0] sum;
    begin
      sum = a + b;
      add_q = sat_q(sum);
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] exp_approx(
      input logic signed [DATA_W-1:0] x
  );
    logic signed [DATA_W-1:0] one_q;
    logic signed [DATA_W-1:0] four_q;
    logic signed [DATA_W-1:0] eight_q;
    logic signed [DATA_W-1:0] x2;
    logic signed [DATA_W-1:0] half_q;
    logic signed [DATA_W-1:0] term2;
    logic signed [DATA_W-1:0] sum;
    begin
      one_q = (1 <<< FRAC_W);
      four_q = (4 <<< FRAC_W);
      eight_q = (8 <<< FRAC_W);
      half_q = (1 <<< (FRAC_W-1));
      if (x < -four_q)
        exp_approx = 0;
      else if (x > four_q)
        exp_approx = eight_q;
      else begin
        x2 = mul_q(x, x);
        term2 = mul_q(x2, half_q);
        sum = add_q(add_q(one_q, x), term2);
        exp_approx = sum;
      end
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] gelu_approx(
      input logic signed [DATA_W-1:0] x
  );
    logic signed [DATA_W-1:0] half_q;
    logic signed [DATA_W-1:0] one_q;
    logic signed [DATA_W-1:0] a_q;
    logic signed [DATA_W-1:0] b_q;
    logic signed [DATA_W-1:0] x2;
    logic signed [DATA_W-1:0] x3;
    logic signed [DATA_W-1:0] t;
    logic signed [DATA_W-1:0] t2;
    logic signed [DATA_W-1:0] tanh_q;
    logic signed [DATA_W-1:0] inner;
    begin
      half_q = (1 <<< (FRAC_W-1));
      one_q = (1 <<< FRAC_W);
      a_q = $rtoi(0.797885 * (1 << FRAC_W));
      b_q = $rtoi(0.044715 * (1 << FRAC_W));

      x2 = mul_q(x, x);
      x3 = mul_q(x2, x);
      t = add_q(x, mul_q(b_q, x3));
      t2 = mul_q(a_q, t);

      if (t2 > one_q)
        tanh_q = one_q;
      else if (t2 < -one_q)
        tanh_q = -one_q;
      else
        tanh_q = t2;

      inner = add_q(one_q, tanh_q);
      gelu_approx = mul_q(mul_q(half_q, x), inner);
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] div_q(
      input logic signed [DATA_W-1:0] num,
      input logic signed [DATA_W-1:0] den
  );
    logic signed [ACC_W-1:0] numerator;
    logic signed [ACC_W-1:0] quotient;
    begin
      if (den == 0)
        div_q = 0;
      else begin
        numerator = num;
        quotient = (numerator <<< FRAC_W) / den;
        div_q = sat_q(quotient);
      end
    end
  endfunction
endpackage

`endif
