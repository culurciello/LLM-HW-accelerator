`ifndef TRANSFORMER_PKG_SV
`define TRANSFORMER_PKG_SV

package transformer_pkg;
  parameter int DATA_W = 16;
  parameter int FRAC_W = 8;
  parameter int ACC_W  = 32;

  function automatic logic signed [DATA_W-1:0] sat_q(
      input logic signed [ACC_W-1:0] v
  );
    logic signed [ACC_W-1:0] max_v;
    logic signed [ACC_W-1:0] min_v;
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
      sum = $signed(a) + $signed(b);
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
      a_q = $signed($rtoi(0.797885 * (1 << FRAC_W)));
      b_q = $signed($rtoi(0.044715 * (1 << FRAC_W)));

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
        numerator = $signed(num);
        quotient = (numerator <<< FRAC_W) / $signed(den);
        div_q = sat_q(quotient);
      end
    end
  endfunction

  function automatic logic [15:0] fp16_mul(
      input logic [15:0] a,
      input logic [15:0] b
  );
    logic sign;
    logic [4:0] exp_a;
    logic [4:0] exp_b;
    logic [5:0] exp_sum;
    logic [10:0] man_a;
    logic [10:0] man_b;
    logic [21:0] man_prod;
    logic [4:0] exp_out;
    logic [9:0] man_out;
    begin
      if (a[14:0] == 0 || b[14:0] == 0) begin
        fp16_mul = 16'h0000;
      end else if (a[14:10] == 5'h1F || b[14:10] == 5'h1F) begin
        fp16_mul = {a[15] ^ b[15], 5'h1F, 10'h000};
      end else begin
        sign = a[15] ^ b[15];
        exp_a = a[14:10];
        exp_b = b[14:10];
        man_a = {1'b1, a[9:0]};
        man_b = {1'b1, b[9:0]};
        man_prod = man_a * man_b;
        exp_sum = exp_a + exp_b - 5'd15;
        if (man_prod[21]) begin
          man_out = man_prod[20:11];
          exp_sum = exp_sum + 1;
        end else begin
          man_out = man_prod[19:10];
        end
        if ($signed(exp_sum) <= 0) begin
          fp16_mul = 16'h0000;
        end else if (exp_sum >= 31) begin
          fp16_mul = {sign, 5'h1F, 10'h000};
        end else begin
          exp_out = exp_sum[4:0];
          fp16_mul = {sign, exp_out, man_out};
        end
      end
    end
  endfunction

  function automatic logic [15:0] fp16_add(
      input logic [15:0] a,
      input logic [15:0] b
  );
    logic sign_a;
    logic sign_b;
    logic [4:0] exp_a;
    logic [4:0] exp_b;
    logic [10:0] man_a;
    logic [10:0] man_b;
    logic [11:0] man_sum;
    logic [4:0] exp_out;
    logic sign_out;
    integer shift;
    begin
      if (a[14:0] == 0)
        fp16_add = b;
      else if (b[14:0] == 0)
        fp16_add = a;
      else if (a[14:10] == 5'h1F)
        fp16_add = a;
      else if (b[14:10] == 5'h1F)
        fp16_add = b;
      else begin
        sign_a = a[15];
        sign_b = b[15];
        exp_a = a[14:10];
        exp_b = b[14:10];
        man_a = {1'b1, a[9:0]};
        man_b = {1'b1, b[9:0]};
        if (exp_a > exp_b) begin
          shift = exp_a - exp_b;
          man_b = (shift >= 11) ? 11'h0 : (man_b >> shift);
          exp_out = exp_a;
        end else begin
          shift = exp_b - exp_a;
          man_a = (shift >= 11) ? 11'h0 : (man_a >> shift);
          exp_out = exp_b;
        end
        if (sign_a == sign_b) begin
          man_sum = man_a + man_b;
          sign_out = sign_a;
        end else if (man_a >= man_b) begin
          man_sum = man_a - man_b;
          sign_out = sign_a;
        end else begin
          man_sum = man_b - man_a;
          sign_out = sign_b;
        end
        if (man_sum == 0) begin
          fp16_add = 16'h0000;
        end else begin
          if (man_sum[11]) begin
            man_sum = man_sum >> 1;
            exp_out = exp_out + 1;
          end else begin
            while (man_sum[10] == 0 && exp_out > 0) begin
              man_sum = man_sum << 1;
              exp_out = exp_out - 1;
            end
          end
          if (exp_out == 0)
            fp16_add = 16'h0000;
          else if (exp_out >= 31)
            fp16_add = {sign_out, 5'h1F, 10'h000};
          else
            fp16_add = {sign_out, exp_out, man_sum[9:0]};
        end
      end
    end
  endfunction
endpackage

`endif
