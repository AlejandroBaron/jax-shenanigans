# RNG

- RNG keys must be completely independent. Otherwise your samples (e.g. Uniform vs Normal) will be correlated [stack overflow issue](https://stackoverflow.com/questions/76135488/jax-random-generator-random-normal-numbers-seem-to-be-returned-sorted-and-not)

# Jit

- Can't use jit on a __call__ method. Jax only admits "primitive" types [see this link](https://github.com/google/jax/issues/4416)
