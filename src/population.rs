use crate::{graph::NeuralNetwork, enecode::EneCode};

/// Population is the struct that contains the agents for selection and mediates the evolutionary
/// algorithm. In this case we will use Stochastic Universal Sampling along with Truncation.
///
///

trait FitnessFunction {
    fn fitness(&self, agent: &NeuralNetwork) -> f32;
}

struct Population {
    pub agents: Vec<NeuralNetwork>,
    pub target_population: usize,
    pub mutation_rate: f32,
    pub generation: usize,
    pub fitness_criterion: f32
}

impl Population {

    pub fn new(genome_base: EneCode, population_size: usize, mutation_rate: f32) -> Self {
        let mut agent_vector: Vec<NeuralNetwork> = Vec::new();

        for idx in 0..population_size {
            let mut agent = NeuralNetwork::new(genome_base.clone());
            agent.initialize();
            agent.mutate(mutation_rate);
            agent_vector.push(agent.transfer());
        }

        Population {
            agents: agent_vector,
            mutation_rate, 
            target_population: population_size,
            generation: 0,
            fitness_criterion: 0.,
        }
    }

    fn evaluate_fitness<T: FitnessFunction>(&self, f: T) -> Vec<f32> {
        let fitness_vector: Vec<f32> = self.agents.iter().map(|x| f.fitness(x)).collect();
        fitness_vector
    }

    fn sample(&self, n_select: usize) -> Vec<usize> {
        Vec::new()
    }

    fn reproduce(&self, a1: NeuralNetwork, a2: NeuralNetwork, n_offspring: usize) -> Vec<NeuralNetwork> {
        Vec::new()
    }

    fn run_generation(&mut self) ->  bool {
        false
    }

}

mod tests {
    use super::*;
    use crate::doctest::GENOME_EXAMPLE;

    #[test]
    fn test_create_population() {
        let genome = GENOME_EXAMPLE.clone();
        let population_test = Population::new(genome, 125, 0.1);
        assert_eq!(population_test.agents.len(), 125);
    }

    #[test]
    fn test_evaluate_fitness() {

        struct TestFitnessObject {
        }

        impl FitnessFunction for TestFitnessObject {
            fn fitness(&self, agent: &NeuralNetwork) -> f32 {
                1.
            }
        }

        let genome = GENOME_EXAMPLE.clone();
        let population_test = Population::new(genome, 125, 0.1);

        let population_fitness = population_test.evaluate_fitness(TestFitnessObject {} );

        assert_eq!(population_fitness.iter().sum::<f32>(), 125.);
    }

    #[test]
    fn test_sample_population() {
    }


    #[test]
    fn test_reproduction() {
    }

    #[test]
    fn test_run_generation() {
    }

}
