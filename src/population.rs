use crate::{graph::NeuralNetwork, enecode::EneCode};

/// Population is the struct that contains the agents for selection and mediates the evolutionary
/// algorithm. In this case we will use Stochastic Universal Sampling along with Truncation.
///
struct population {
    pub agents: Vec<NeuralNetwork>,
    pub target_population: usize,
    pub mutation_rate: f32,
    pub generation: usize,
    pub fitness_criterion: f32
}

impl population {

    pub fn new(genome_base: EneCode, population_size: usize, mutation_rate: f32) -> Self {
        let mut agent_vector: Vec<NeuralNetwork> = Vec::new();

        for idx in 0..population_size {
            let mut agent = NeuralNetwork::new(genome_base.clone());
            agent.initialize();
            agent.mutate(mutation_rate);
            agent_vector.push(agent.transfer());
        }

        population {
            agents: agent_vector,
            mutation_rate, 
            target_population: population_size,
            generation: 0,
            fitness_criterion: 0.,
        }
    }

    fn evaluate_fitness(&self) -> Vec<f32> {
        Vec::new()
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

    #[test]
    fn test_create_population() {
    }

    #[test]
    fn test_evaluate_fitness() {
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
