#  Evo RL

Evo RL is a machine learning library built in Rust to explore the evolution strategies for the creation of artificial neural networks. Neural Networks are implemented as graphs specified by a direct encoding scheme, which allows crossover during selection. 

## Neuroevolution

[Neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution) is a field in artificial intelligence which leverages evolutionary algorithms to create structured artificial neural networks. 

The main evolutionary algorithm in this libary is inspired by the [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) (K.O. Stanley and  R. Miikkulainen) and implements stochastic universal sampling with truncation as the selection mechanism. 

A survey/discussion of recent advances and other packages in this area as of 2024 can be found in [this paper](https://arxiv.org/abs/2303.04150). 

Alternatively, [EvoJAX](https://github.com/google/evojax) presents a more complete and scalable toolkit which implements many neuroevolution algorithms.

## Python
A python package (evo_rl) can be built by running `maturin develop` in the source code. Examples are included in the `examples` directory. 

A code snippet is reproduced here:

```
#A Python script which trains an agent to solve the mountain car task in OpenAI's Gymnasium

import evo_rl
import logging

from utils import MountainCarEnvironment, visualize_gen

import gymnasium as gym
import numpy as np

FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

population_size = 200

configuration = {
        "population_size": population_size,
        "survival_rate": 0.2,
        "mutation_rate": 0.4, 
        "input_size": 2,
        "output_size": 2,
        "topology_mutation_rate": 0.4,
        "project_name": "mountaincar",
        "project_directory": "mc_agents"
        }

env = gym.make('MountainCarContinuous-v0')
mc = MountainCarEnvironment(env, configuration)

p = evo_rl.PopulationApi(configuration)

while p.generation < 1000:

    for agent in range(population_size):
        mc.evaluate_agent(p, agent)

    if p.fitness > 100:
        break
        
    p.update_population_fitness()
    p.report()
    p.evolve_step()

```

## Running Tests

### Verbose
 RUST_LOG=[debug/info] cargo test -- --nocapture

