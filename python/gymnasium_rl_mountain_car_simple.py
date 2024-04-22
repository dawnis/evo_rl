#A script which trains an agent to solve the mountain car task in OpenAI's Gymnasium

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

#visualize_gen("mc_agents/mountaincar_0200")
#mc.observe(p, 128)
