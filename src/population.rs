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
    pub fitness_criterion: f32,
    survival_rate: f32,
    agent_fitness: Vec<f32>,
}

impl Population {

    pub fn new(genome_base: EneCode, population_size: usize, mutation_rate: f32) -> Self {
        let mut agent_vector: Vec<NeuralNetwork> = Vec::new();

        for _idx in 0..population_size {
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
            survival_rate: 0.8,
            agent_fitness: Vec::new(),
        }
    }

    fn evaluate_fitness<T: FitnessFunction>(&mut self, f: T) {
        let fitness_vector: Vec<f32> = self.agents.iter().map(|x| f.fitness(x)).collect();
        self.agent_fitness = fitness_vector;
    }

    fn selection(&self, n_select: usize) -> Vec<NeuralNetwork> {
        let truncated_population = self.truncate_population();
        self.stochastic_universal_sampling(truncated_population)
    }

    fn stochastic_universal_sampling(&self, sample: Vec<usize>) -> Vec<NeuralNetwork> {
        Vec::new()
    }

    ///Returns indices associated with truncated population
    fn truncate_population(&self) -> Vec<usize> {
        let n_survival = (self.agents.len() as f32) * self.survival_rate;
        let mut fitness_pairing: Vec<_> = (0..self.agents.len()).zip(&self.agent_fitness).collect();
        fitness_pairing.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let sorted_data: Vec<usize> = fitness_pairing.into_iter().map(|(a, _)| a).collect();

        sorted_data[..n_survival.round() as usize].to_vec()

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
            fn fitness(&self, _agent: &NeuralNetwork) -> f32 {
                1.
            }
        }

        let genome = GENOME_EXAMPLE.clone();
        let mut population_test = Population::new(genome, 125, 0.1);

        population_test.evaluate_fitness(TestFitnessObject {} );

        assert_eq!(population_test.agent_fitness.iter().sum::<f32>(), 125.);
    }

    #[test]
    fn test_truncate_population() {
        let genome = GENOME_EXAMPLE.clone();
        let mut population_test = Population::new(genome, 10, 0.1);
        population_test.agent_fitness = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        
        let sorted_fitness_indices = population_test.truncate_population();

        assert_eq!(sorted_fitness_indices, vec![9, 8, 7, 6, 5, 4, 3, 2]);
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
