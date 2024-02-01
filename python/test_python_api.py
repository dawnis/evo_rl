import evo_rl

configuration = {
        "population_size": 200,
        "survival_rate": 0.2,
        "mutation_rate": 0.4, 
        "topology_mutation_rate": 0.4
        }

p = evo_rl.PopulationApi(configuration)

p.evolve()
