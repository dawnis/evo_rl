#  Evo RL

Evo RL is a machine learning library built in Rust to explore the evolution of artificial biomemetic neural networks. Neural Networks are implemented as graphs specified by a direct encoding scheme, which allows crossover during selection. 

## Evolutionary Algorithm

The main evolutionary algorithm in this libary is inspired by the NEAT (K.O. Stanley and  R. Miikkulainen) and implements Stochastic Universal Sampling with Truncation as the selection mechanism. 

## Python
An python package (evo_rl) can be built by running `maturin develop` in the source code. Examples are included in the `python` directory. 

## Running Tests

### Verbose
 RUST_LOG=[debug/info] cargo test -- --nocapture

