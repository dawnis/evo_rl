from graphviz import Source
from IPython.display import display
from scipy.special import expit

import gymnasium as gym
import numpy as np
import random

def visualize_gen(filename):
    """Visualize the graph for a particular generation"""
    # Path to your .dot file
    file_path = f"{filename}.dot"
    source = Source.from_file(file_path)
    display(source)
    output_path = source.render(filename=file_path, format='png')
    return file_path

def bound_discrete(x):
    """transforms a continuous network output into either 0 or 1 using expit function from scipy"""
    expit_x = expit(x)
    action = expit_x >= 0.5
    return int(action)


class GymnasiumEnv:

    def __init__(self, environment_name, configuration): 
        self.env_name = environment_name
        self.env = gym.make(environment_name)
        self.evaluation_steps = 999
        self.observation_steps = 999
        self.configuration = configuration

    def bound_action(self, action_raw):
        """Bounds output action to action space"""
        return action_raw

    def observe(self, population, idx):
        self.env = gym.make(self.env_name, render_mode="human")
        agent_idx = idx
        observation, info = self.env.reset()

        for _ in range(self.evaluation_steps):
            population.agent_fwd(agent_idx, list(observation))
            action_raw = population.agent_out(agent_idx)
            action = self.bound_action(action_raw)
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()

    def write_agent(self, population, agent_idx, file_save_path):
        population.agent_checkpt(agent_idx, file_save_path)

class LunarLanderEnv(GymnasiumEnv):

    def __init__(self, configuration): 
        super().__init__("LunarLander-v2", configuration)

    def bound_action(self, action_raw):
        action_value = action_raw[0]

        if action_value == 0:
            return 0
        elif action_value < -0.5:
            return 1
        elif action_value > 0.5:
            return 3
        else: 
            return 2
        
    def evaluate_agent(self, population, agent_idx):
        fitness = 0
        observation, info = self.env.reset()

        for _ in range(self.evaluation_steps):
            population.agent_fwd(agent_idx, list(observation))
            action_raw = population.agent_out(agent_idx)
            action = self.bound_action(action_raw)
            observation, reward, terminated, truncated, info = self.env.step(action)

            fitness += reward

            if fitness != fitness:
                print(f"Error! Observing NaN Fitness")

            if terminated or truncated:
                observation, info = self.env.reset()

        fitness -= 0.1 * population.agent_complexity(agent_idx)
        
        if fitness < 0:
            population.set_agent_fitness(agent_idx, random.uniform(0, 1))
        else:
            population.set_agent_fitness(agent_idx, fitness)

class CartPoleEnvironment(GymnasiumEnv):

    def __init__(self, configuration): 
        super().__init__("CartPole-v1", configuration)

    def bound_action(self, action_raw):
        """Bounds action output to acceptable action space"""
        return bound_discrete(action_raw[0])

    def evaluate_agent(self, population, agent_idx):
        observation, info = self.env.reset()
        fitness = 0
        termination_penalty = min([100 * population.generation + 1, 1000])
        step = 0

        for _ in range(self.evaluation_steps):
            step += 1
            population.agent_fwd(agent_idx, list(observation))
            action_raw = population.agent_out(agent_idx)
            action = self.bound_action(action_raw)
            observation, reward, terminated, truncated, info = self.env.step(action)

            reward_delta = reward #sometimes there are NaNs, which crash the program

            if (reward_delta == reward_delta) & (step > 9):
                fitness += reward_delta #forward progress + clipped reward penalty - complexity penalty

            if fitness != fitness:
                print(f"Error! Observing NaN Fitness")

            if terminated or truncated:
                #penalize every termination with a constant penalty
                step = 0
                new_fitness = fitness - termination_penalty
                fitness = max([new_fitness, 0])

                observation, info = self.env.reset()

        #print(f"Setting agent {agent_idx} fitness to {fitness}")

        fitness -= 0.1 * population.agent_complexity(agent_idx)
        
        if fitness < 0:
            population.set_agent_fitness(agent_idx, random.uniform(0, 1))
        else:
            population.set_agent_fitness(agent_idx, fitness)

    
class MountainCarEnvironment(GymnasiumEnv):

    def __init__(self, configuration): 
        super().__init__("MountainCarContinuous-v0", configuration)

    def evaluate_agent(self, population, agent_idx):
        observation, info = self.env.reset()
        fitness = 0

        for _ in range(self.evaluation_steps):
            population.agent_fwd(agent_idx, list(observation))
            action = population.agent_out(agent_idx)
            observation, reward, terminated, truncated, info = self.env.step(action)

            reward_delta = max([0, observation[0]])*10 + reward #sometimes there are NaNs, which crash the program

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

    
