import evo_rl
import logging
from evo_rl import FitnessEvaluator


FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

configuration = {
        "population_size": 200,
        "survival_rate": 0.2,
        "mutation_rate": 0.4, 
        "topology_mutation_rate": 0.4
        }


evo_rl.log_something()

@FitnessEvaluator
def lambda_n(agent):
    f = agent.fwd((0,1))
    print(f"Hello, I am Mr. Fitness Evaluator, with fitness {f}")
    return f

p = evo_rl.PopulationApi(configuration, lambda_n)

p.evolve()
