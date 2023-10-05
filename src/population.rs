use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

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

    fn selection(&self, n_select: usize) -> Vec<usize> {
        let truncated_population = self.truncate_population();
        self.stochastic_universal_sampling(truncated_population, n_select)
    }

    /// Implements stochastic universal sampling, an efficient algorithm related to Roulette Wheel
    /// Selection for evolutionary aglorithms
    fn stochastic_universal_sampling(&self, sample: Vec<usize>, n_select: usize) -> Vec<usize> {
        let sample_fitness: Vec<f32> = sample.iter().map(|&idx| self.agent_fitness[idx]).collect();
        let total_population_fitness: f32 = sample_fitness.iter().sum();
        let point_spacing = total_population_fitness / (n_select as f32);
        let mut rng = rand::thread_rng();
        let u = Uniform::from(0_f32..point_spacing);

        let mut selection: Vec<usize>  = Vec::new();

        //start the roulette pointer at a point between 0 and the first spacing
        let mut roulette_pointer = u.sample(&mut rng); 

        for _p in 0..n_select {
            let mut idx: usize = 0;
            while sample_fitness[0..idx].iter().sum::<f32>() < roulette_pointer {
                idx += 1;
            }
            selection.push(sample[idx-1]);
            roulette_pointer += point_spacing;
        }

        selection
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
    fn test_stochastic_universal_sampling() {
        let genome = GENOME_EXAMPLE.clone();
        let mut population_test = Population::new(genome, 3, 0.1);
        population_test.agent_fitness = vec![5., 3., 2.];
        let sample: Vec<usize> = vec![0, 1, 2];

        let sus: Vec<usize> = population_test.stochastic_universal_sampling(sample, 10);
        assert_eq!(vec![0, 0, 0, 0, 0, 1, 1, 1, 2, 2], sus);
    }


    #[test]
    fn test_reproduction() {
    }

    #[test]
    fn test_run_generation() {
    }

}
