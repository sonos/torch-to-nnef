FRAGMENTS = {
    "silu": """
fragment silu( input: tensor<scalar> ) -> ( output : tensor<scalar> )
{
  output = input * sigmoid(input);
}
""".strip(),
    "selu": """
fragment selu(
  x: tensor<scalar>,
  alpha: scalar = 1.67326319,
  lambda: scalar = 1.05070102 )
-> ( y: tensor<scalar> )
{
    y = lambda * select(x < 0.0, alpha * (exp(x) - 1.0), x);
}
""".strip(),
    "gelu": """
fragment gelu( x: tensor<scalar> ) -> ( y: tensor<scalar> )
{
    y = 0.5 * x * (1 + tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)));
}
""".strip(),
}
