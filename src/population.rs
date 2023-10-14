use rand::Rng;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use log::*;

use crate::{graph::NeuralNetwork, enecode::EneCode};

/// Population is the struct that contains the agents for selection and mediates the evolutionary
/// algorithm. In this case we will use Stochastic Universal Sampling along with Truncation.
///
///

trait FitnessEvaluation {
    fn fitness(&self, agent: &mut NeuralNetwork) -> f32;
}

///Configuration for Population to Set Hyperparameters
struct PopulationConfig<F: FitnessEvaluation> {
    evaluator: F, 
    rng: Box<dyn RngCore>,
    epoch_size: usize,
    mutation_rate_scale_per_epoch: f32,
    mutation_effect_scale_per_epoch: f32,
}

impl<F: FitnessEvaluation> PopulationConfig<F> {
    pub fn new(evaluator: F, 
               epoch_size: usize, 
               mutation_rate_scale_per_epoch: f32, 
               mutation_effect_scale_per_epoch: f32,
               rng_seed: Option<u8>) -> Self {

        let mut rng: Box<dyn RngCore> = match rng_seed {
            Some(seedu8) => {
                let seed = [seedu8; 32];
                Box::new(StdRng::from_seed(seed))
            }
            None => Box::new(rand::thread_rng())
        };

        PopulationConfig {
            evaluator,
            rng,
            epoch_size,
            mutation_rate_scale_per_epoch,
            mutation_effect_scale_per_epoch,
        }
    }
}

struct Population {
    pub agents: Vec<NeuralNetwork>,
    pub size: usize,
    pub mutation_rate: f32,
    pub mutation_effect_sd: f32,
    pub generation: usize,
    pub population_fitness: f32,
    survival_rate: f32,
    agent_fitness: Vec<f32>,
}

impl Population {

    pub fn new(genome_base: EneCode, population_size: usize, survival_rate: f32, mutation_rate: f32) -> Self {
        let mut agent_vector: Vec<NeuralNetwork> = Vec::new();

        for _idx in 0..population_size {
            let mut agent = NeuralNetwork::new(genome_base.clone());
            agent.initialize();
            // Random initialization of the population of all parameters
            agent.mutate(1., 10.);
            agent_vector.push(agent.transfer());
        }

        Population {
            agents: agent_vector,
            mutation_rate, 
            mutation_effect_sd: 5.,
            size: population_size,
            generation: 0,
            population_fitness: 0.,
            survival_rate,
            agent_fitness: Vec::new(),
        }
    }

    fn evaluate_fitness<T: FitnessEvaluation>(&mut self, f: &T) {
        let fitness_vector: Vec<f32> = self.agents.iter_mut().map(|x| f.fitness(x)).collect();
        self.agent_fitness = fitness_vector;
    }

    fn selection<R: Rng>(&self, rng: &mut R, n_select: usize) -> Vec<usize> {
        let truncated_population = self.truncate_population();
        self.stochastic_universal_sampling(rng, truncated_population, n_select)
    }

    /// Implements stochastic universal sampling, an efficient algorithm related to Roulette Wheel
    /// Selection for evolutionary aglorithms
    fn stochastic_universal_sampling<R: Rng>(&self, rng: &mut R, sample: Vec<usize>, n_select: usize) -> Vec<usize> {
        let sample_fitness: Vec<f32> = sample.iter().map(|&idx| self.agent_fitness[idx]).collect();
        let total_population_fitness: f32 = sample_fitness.iter().sum();
        let point_spacing = total_population_fitness / (n_select as f32);
        let u = Uniform::from(0_f32..point_spacing);

        let mut selection: Vec<usize>  = Vec::new();

        //start the roulette pointer at a point between 0 and the first spacing
        let mut roulette_pointer = u.sample(rng); 

        for _p in 0..n_select {
            let mut idx: usize = 0;
            while sample_fitness[0..idx].iter().sum::<f32>() < roulette_pointer {
                idx += 1;

                if idx == sample_fitness.len() {
                    break;
                }
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

    fn generate_offspring<R: Rng>(&self, rng: &mut R, parental_ids: Vec<usize>) -> Vec<NeuralNetwork> {
        let mut offspring: Vec<NeuralNetwork> = Vec::new();

        // Given selected parents, mate in pairs until the population size is fulfilled
        let num_parents = parental_ids.len();

        let mate_attempt_limit = self.size as f32 * 1.5;

        let mut n_mate_attempts = 0;

        while offspring.len() < self.size {
            let a1 = rng.gen_range(0..num_parents);
            let partner_list: Vec<&usize> = parental_ids.iter().filter(|&&p| p != a1).collect();

            let a2 = rng.gen_range(0..partner_list.len());
            let parent_1 = parental_ids[a1];
            let parent_2 = *partner_list[a2];

            let offspring_nn = self.agents[parent_1].recombine_enecode(rng, &self.agents[parent_2] );

            match offspring_nn {
                Ok(nn) => offspring.push(nn),
                Err(e) => println!("Recombination failed: {:#?}", e),
            }

            n_mate_attempts += 1;

            if n_mate_attempts > mate_attempt_limit as i32 {
                panic!("Offspring mating has exceeded 50% failure rate.");
            }
        }

        offspring
    }

    pub fn evolve<F: FitnessEvaluation>(&mut self, pop_config: PopulationConfig<F>, iterations_max: usize, max_fitness_criterion: f32) {
        let mut rng = pop_config.rng;

        //reset generation value
        self.generation = 0;

        while self.generation < iterations_max {
            self.evaluate_fitness(&pop_config.evaluator);

            // Select same population size, but use SUS to select according to fitness
            let selection = self.selection(&mut rng, self.size);

            let mut offspring = self.generate_offspring(&mut rng, selection);

            for agent in offspring.iter_mut() {
                agent.mutate(self.mutation_rate, self.mutation_effect_sd);
            }

            self.agents = offspring;
            self.population_fitness = self.agent_fitness.iter().sum::<f32>() / self.size as f32;
            self.generation += 1;

            let population_max = self.agent_fitness.clone().into_iter().reduce(f32::max).unwrap();

            if self.population_fitness > max_fitness_criterion {
                break;
            }

            if self.generation % pop_config.epoch_size == 0 {
                self.mutation_rate *= pop_config.mutation_rate_scale_per_epoch;
                self.mutation_effect_sd *= pop_config.mutation_effect_scale_per_epoch;
            }

            info!("Observing population fitness {} on generation {} with max of {}", self.population_fitness, self.generation, population_max);

        }

    }

}

mod tests {
    use super::*;
    use crate::graph::NeuralNetwork;
    use crate::doctest::{GENOME_EXAMPLE, XOR_GENOME};
    use crate::setup_logger;

    #[test]
    fn test_create_population() {
        let genome = GENOME_EXAMPLE.clone();
        let population_test = Population::new(genome, 125, 0.8, 0.1);
        assert_eq!(population_test.agents.len(), 125);
    }

    #[test]
    fn test_evaluate_fitness() {

        struct TestFitnessObject {
        }

        impl FitnessEvaluation for TestFitnessObject {
            fn fitness(&self, _agent: &mut NeuralNetwork) -> f32 {
                1.
            }
        }

        let genome = GENOME_EXAMPLE.clone();
        let mut population_test = Population::new(genome, 125, 0.8, 0.1);

        population_test.evaluate_fitness(&TestFitnessObject {} );

        assert_eq!(population_test.agent_fitness.iter().sum::<f32>(), 125.);
    }

    #[test]
    fn test_truncate_population() {
        let genome = GENOME_EXAMPLE.clone();
        let mut population_test = Population::new(genome, 10, 0.8, 0.1);
        population_test.agent_fitness = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        
        let sorted_fitness_indices = population_test.truncate_population();

        assert_eq!(sorted_fitness_indices, vec![9, 8, 7, 6, 5, 4, 3, 2]);
    }

    #[test]
    fn test_stochastic_universal_sampling() {
        let seed = [17; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);

        let genome = GENOME_EXAMPLE.clone();
        let mut population_test = Population::new(genome, 3, 0.8, 0.1);
        population_test.agent_fitness = vec![5., 3., 2.];
        let sample: Vec<usize> = vec![0, 1, 2];

        let sus: Vec<usize> = population_test.stochastic_universal_sampling(&mut rng, sample, 10);
        assert_eq!(vec![0, 0, 0, 0, 0, 1, 1, 1, 2, 2], sus);
    }


    #[test]
    fn test_generate_offspring() {
        let seed = [17; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);

        let genome = GENOME_EXAMPLE.clone();
        let population_test = Population::new(genome, 10, 0.8, 0.1);
        let parent_id_vector = vec![0, 1, 3, 5];

        let offspring_vec = population_test.generate_offspring(&mut rng, parent_id_vector);

        assert_eq!(offspring_vec.len(), 10);
    }

    #[test]
    fn test_evolve() {
        setup_logger();

        let genome = XOR_GENOME.clone();
        let mut population = Population::new(genome, 200, 0.1, 0.4);

        struct XorEvaluation {
            pub fitness_begin: f32
        }

        impl XorEvaluation {
            pub fn new() -> Self {
                XorEvaluation {
                    fitness_begin: 4.0
                }
            }

        }

        impl FitnessEvaluation for XorEvaluation {
            fn fitness(&self, agent: &mut NeuralNetwork) -> f32 {
                let mut fitness_evaluation = self.fitness_begin;

                for bit1 in 0..2 {
                    for bit2 in 0..2 {
                        agent.fwd(vec![bit1 as f32, bit2 as f32]);
                        let network_output = agent.fetch_network_output();

                        let xor_true = (bit1 > 0) ^ (bit2 > 0);
                        let xor_true_float: f32 = if xor_true {1.} else {0.};

                        fitness_evaluation -= (xor_true_float - network_output[0]).powf(2.);

                    }
                }

                fitness_evaluation
            }
        }
        let ef = XorEvaluation::new();
        
        let config = PopulationConfig::new(ef, 50, 0.50, 0.50, Some(13));

        population.evolve(config, 400, 3.8);
        assert!(population.population_fitness >= 3.7);
    }

}
