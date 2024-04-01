import evo_rl
import logging

FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

configuration = {
        "population_size": 200,
        "survival_rate": 0.2,
        "mutation_rate": 0.4, 
        "topology_mutation_rate": 0.4
        }


def evaluate_xor_fitness(agent):
    fitness = 6

    #TODO: Complexity Penalty
    for bit1 in [0, 1]:
        for bit2 in [0, 2]:
            agent.fwd([bit1, bit2])
            network_output = agent.nn.fetch_network_output()
            xor_true = (bit1 > 0) ^ (bit2 > 0)
            fitness -= (xor_true - network_output)**2

    agent.update(fitness)
    return 

evo_rl.log_something()

p = evo_rl.PopulationApi(configuration)

while p.current_generation() < 1000:

    if p.fitness() >= 5.8:
        break

    #TODO: Iterate and evaluate over agents
    for agent in p.agents: 
        evaluate_xor_fitness(agent)

    p.update_population_fitness()
    p.report()
    p.evolve_step()
