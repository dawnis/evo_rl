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
        "input_size": 2,
        "output_size": 2,
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

#example code for checkpointing a particular agent (51) and starting a new population with the checkpt genome
mc.write_agent(p, 51, "path/to/agents/mc_agents/agent_checkpt.json")

#start new population with same config but checkpt genome
v = evo_rl.PopulationApi(configuration, "path/to/agents/mc_agents/agent_checkpt.json")

#re-run same task with new popualtion -- should take fewer generations
while v.generation < 1000:

    for agent in range(population_size):
        mc.evaluate_agent(v, agent)

    if v.fitness > 100:
        break
        
    v.update_population_fitness()
    v.report()
    v.evolve_step()
