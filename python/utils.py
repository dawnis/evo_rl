from graphviz import Source
from IPython.display import display

import gymnasium as gym
import random

def visualize_gen(filename):
    """Visualize the graph for a particular generation"""
    # Path to your .dot file
    file_path = f"{filename}.dot"
    source = Source.from_file(file_path)
    display(source)
    output_path = source.render(filename=file_path, format='png')
    return file_path

class MountainCarEnvironment:

    def __init__(self, environment, configuration): 
        self.env = environment
        self.evaluation_steps = 200
        self.configuration = configuration


    def evaluate_agent(self, population, agent_idx):
        observation, info = self.env.reset()
        fitness = 0

        for _ in range(self.evaluation_steps):
            population.agent_fwd(agent_idx, [observation[0], observation[1]])
            action = population.agent_out(agent_idx)
            observation, reward, terminated, truncated, info = self.env.step(action)

            reward_delta = max([0, observation[1]]) + reward #sometimes there are NaNs, which crash the program

            if reward_delta == reward_delta:
                fitness += reward_delta #forward progress + clipped reward penalty - complexity penalty

            if fitness != fitness:
                print(f"Error! Observing NaN Fitness")
                
            if terminated or truncated:
                observation, info = self.env.reset()

        #print(f"Setting agent {agent_idx} fitness to {fitness}")

        fitness -= 0.1 * population.agent_complexity(agent_idx)
        
        if fitness < 0:
            population.set_agent_fitness(agent_idx, random.uniform(0, 1))
        else:
            population.set_agent_fitness(agent_idx, fitness)

    
    def observe(self, population, idx):
        self.env = gym.make('MountainCarContinuous-v0', render_mode="human")
        agent_idx = idx
        observation, info = self.env.reset()

        for _ in range(self.evaluation_steps):
            population.agent_fwd(agent_idx, [observation[0], observation[1]]) # use random agent in population for now
            action = population.agent_out(agent_idx)
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()
