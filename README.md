#  Evo RL

Evo RL is a machine learning library built in Rust to explore the evolution strategies for the creation of artificial neural networks. Neural Networks are implemented as graphs specified by a direct encoding scheme, which allows crossover during selection. 

## Neuroevolution

[Neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution) is a field in artificial intelligence which leverages evolutionary algorithms to create structured artificial neural networks. 

The main evolutionary algorithm in this libary is inspired by the [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) (K.O. Stanley and  R. Miikkulainen) and implements stochastic universal sampling with truncation as the selection mechanism. 

A survey/discussion of recent advances and other packages in this area as of 2024 can be found in [this paper](https://arxiv.org/abs/2303.04150). 

Alternatively, [EvoJAX](https://github.com/google/evojax) presents a more complete and scalable toolkit which implements many neuroevolution algorithms.


## Python
A python package (evo_rl) can be built by running `maturin develop` in the source code. Examples are included in the `python` directory. 

## Running Tests

### Verbose
 RUST_LOG=[debug/info] cargo test -- --nocapture

