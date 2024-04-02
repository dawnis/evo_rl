import evo_rl
import logging

FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

population_size = 200

configuration = {
        "population_size": population_size,
        "survival_rate": 0.2,
        "mutation_rate": 0.4, 
        "topology_mutation_rate": 0.4
        }


def evaluate_xor_fitness(population_api, agent_idx):
    fitness = 6

    complexity = population_api.agent_complexity(agent_idx)
    complexity_penalty = 0.1 * complexity

    for bit1 in [0, 1]:
        for bit2 in [0, 2]:
            population_api.agent_fwd(agent_idx, [bit1, bit2])
            network_output = population_api.agent_out(agent_idx)
            xor_true = (bit1 > 0) ^ (bit2 > 0)
            fitness -= (xor_true - network_output[0])**2

    if fitness > complexity_penalty:
        fitness_eval = fitness - complexity_penalty
    else:
        fitness_eval = 0

    population_api.set_agent_fitness(agent_idx, fitness_eval)
    return 

evo_rl.log_something()

p = evo_rl.PopulationApi(configuration)

while p.generation < 1000:

    if p.fitness >= 5.8:
        break

    for agent in range(population_size):
        evaluate_xor_fitness(p, agent)

    p.update_population_fitness()
    p.report()
    p.evolve_step()
