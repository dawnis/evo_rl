use crate::{graph::FeedForwardNeuralNetwork, enecode::EneCode};

/// Population is the struct that contains the agents for selection and mediates the evolutionary
/// algorithm. In this case we will use Stochastic Universal Sampling along with Truncation.
///
struct population {
    pub agents: Vec<FeedForwardNeuralNetwork>,
    pub target_population: usize,
    pub generation: usize,
    pub fitness_criterion: f32
}

impl population {

    fn new(genome_configuration: EneCode, population_size: usize) -> Self {
        population {
            agents: Vec::new(),
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

    fn reproduce(&self, a1: FeedForwardNeuralNetwork, a2: FeedForwardNeuralNetwork, n_offspring: usize) -> Vec<FeedForwardNeuralNetwork> {
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
