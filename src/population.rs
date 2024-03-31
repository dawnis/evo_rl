//!This module focuses on implementing an evolutionary algorithm for neural network optimization. It uses Stochastic Universal Sampling (SUS) and Truncation for selection within a population of neural network agents.

use rand::Rng;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use log::*;
use thiserror::Error;
use std::sync::Arc;

use crate::agent_wrapper::*;
use crate::{graph::NeuralNetwork, enecode::EneCode};

/// `PopulationConfig` is a struct that configures `Population` for evolutionary selection.
///- **Purpose**: Configuration struct for setting hyperparameters of a population.
///- **Fields**:
///  - `evaluator`: A fitness evaluator of type `F` that implements `FitnessEvaluation`.
///  - `rng`: A boxed random number generator.
///  - `epoch_size`: Size of an epoch.
///  - `mutation_rate_scale_per_epoch`: Scaling factor for mutation rate per epoch.
///  - `mutation_effect_scale_per_epoch`: Scaling factor for the effect of mutation per epoch.
///- **Method `new`**: Constructs a new `PopulationConfig`.
///- **Parameters**: 
///  - `evaluator`, `epoch_size`, `mutation_rate_scale_per_epoch`, `mutation_effect_scale_per_epoch`, `rng_seed`.
pub struct PopulationConfig {
    project_name: String, 
    project_directory: String,
    rng: Box<dyn RngCore>,
    epoch_size: usize,
    mutation_rate_scale_per_epoch: f32,
    mutation_effect_scale_per_epoch: f32,
    visualize_best_agent: bool,
}

#[derive(Debug, Error)]
pub enum FitnessValueError{
    #[error("Negative fitness values are not allowed!")]
    NegativeFitnessError
}

impl PopulationConfig {
    pub fn new(project_name: String,
               home_directory: Option<String>,
               epoch_size: usize, 
               mutation_rate_scale_per_epoch: f32, 
               mutation_effect_scale_per_epoch: f32,
               visualize_best_agent: bool,
               rng_seed: Option<u8>) -> Self {

        let mut rng: Box<dyn RngCore> = match rng_seed {
            Some(seedu8) => {
                let seed = [seedu8; 32];
                Box::new(StdRng::from_seed(seed))
            }
            None => Box::new(rand::thread_rng())
        };

        let project_directory: String = match home_directory {
            Some(dir) => dir,
            None => String::from(""),
        };

        PopulationConfig {
            project_name,
            project_directory,
            rng,
            epoch_size,
            mutation_rate_scale_per_epoch,
            mutation_effect_scale_per_epoch,
            visualize_best_agent,
        }
    }
}

///### `Population`
///- **Purpose**: Represents a population in the evolutionary algorithm.
///- **Fields**:
///  - `agents`: A vector of `NeuralNetwork`.
///  - `size`: Size of the population.
///  - `topology_mutation_rate`, `mutation_rate`, `mutation_effect_sd`: Parameters for mutation.
///  - `generation`: Current generation number.
///  - `population_fitness`: Average fitness of the population.
///  - `survival_rate`: Rate at which agents survive per generation.
///  - `agent_fitness`: Vector storing fitness of each agent.
///- **Method `new`**: Constructs a new `Population`.
///- **Parameters**: 
///  - `genome_base`, `population_size`, `survival_rate`, `mutation_rate`, `topology_mutation_rate`.
pub struct Population {
    pub agents: Vec<Agent>,
    pub size: usize,
    pub topology_mutation_rate: f32,
    pub mutation_rate: f32,
    pub mutation_effect_sd: f32,
    pub generation: usize,
    pub population_fitness: f32,
    survival_rate: f32,
    agent_fitness: Vec<f32>,
}

impl Population {

    pub fn new(genome_base: EneCode, population_size: usize, survival_rate: f32, mutation_rate: f32, topology_mutation_rate: f32) -> Self {
        let mut agent_vector:Vec<Agent> = Vec::new();

        for _idx in 0..population_size {
            let agent = Agent::new(genome_base.clone());
            agent_vector.push(agent);
        }

        Population {
            agents: agent_vector,
            topology_mutation_rate,
            mutation_rate, 
            mutation_effect_sd: 5.,
            size: population_size,
            generation: 0,
            population_fitness: 0.,
            survival_rate,
            agent_fitness: Vec::new(),
        }
    }

    ///### `selection`
    ///- **Purpose**: Selects a subset of agents from the population for reproduction.
    ///- **Parameters**:
    ///  - `rng`: A mutable reference to a random number generator.
    ///  - `n_select`: Number of agents to select.
    fn selection<R: Rng>(&self, rng: &mut R, n_select: usize) -> Vec<usize> {
        let truncated_population = self.truncate_population();
        self.stochastic_universal_sampling(rng, truncated_population, n_select)
    }

    ///### `stochastic_universal_sampling`
    ///- **Purpose**: Implements SUS for efficient selection in evolutionary algorithms.
    ///- **Parameters**:
    ///  - `rng`, `sample`, `n_select`.
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

    ///### `truncate_population`
    ///- **Purpose**: Truncates the population based on survival rate.
    ///- **Returns**: A vector of indices representing the surviving population.
    fn truncate_population(&self) -> Vec<usize> {
        let n_survival = (self.agents.len() as f32) * self.survival_rate;
        let mut fitness_pairing: Vec<_> = (0..self.agents.len()).zip(&self.agent_fitness).collect();
        fitness_pairing.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let sorted_data: Vec<usize> = fitness_pairing.into_iter().map(|(a, _)| a).collect();

        sorted_data[..n_survival.round() as usize].to_vec()

    }

    ///### `generate_offspring`
    ///- **Purpose**: Generates offspring from selected parents.
    ///- **Parameters**:
    ///  - `rng`: A mutable reference to a random number generator.
    ///  - `parental_ids`: Vector of indices representing selected parents.
    fn generate_offspring<R: Rng>(&self, rng: &mut R, parental_ids: Vec<usize>) -> Vec<Agent> {
        let mut offspring: Vec<Agent> = Vec::new();

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

            let offspring_ec = self.agents[parent_1].nn.recombine_enecode(rng, &self.agents[parent_2].nn );

            match offspring_ec {
                Ok(ec) => {
                    let agent = Agent::new(ec);
                    offspring.push(agent);
                }
                Err(e) => debug!("Recombination failed: {:#?}", e),
            }

            n_mate_attempts += 1;

            if n_mate_attempts > mate_attempt_limit as i32 {
                panic!("Offspring mating has exceeded 50% failure rate.");
            }
        }

        offspring
    }

    ///### `evolve`
    ///- **Purpose**: Evolves the population over a number of generations.
    ///- **Parameters**:
    ///  - `pop_config`: Population configuration.
    ///  - `iterations_max`: Maximum number of iterations.
    ///  - `max_fitness_criterion`: Fitness threshold to halt evolution.
    pub fn evolve_step(&mut self, pop_config: PopulationConfig) {
        let mut rng = pop_config.rng;

        // Select same population size, but use SUS to select according to fitness
        let selection = self.selection(&mut rng, self.size);

        let mut offspring = self.generate_offspring(&mut rng, selection);

        for agent in offspring.iter_mut() {
            agent.nn.mutate(self.mutation_rate, self.mutation_effect_sd, self.topology_mutation_rate);
        }

        self.agents = offspring;
        self.population_fitness = self.agent_fitness.iter().sum::<f32>() / self.size as f32;
        self.generation += 1;

        let (best_agent_idx, population_max) = self.agent_fitness
                                                   .clone()
                                                   .into_iter()
                                                   .enumerate()
                                                   .fold((0, std::f32::MIN), |(idx_max, val_max), (idx, val) | {
                                                        if val > val_max { (idx, val) } else { (idx_max, val_max) }
                                                   });

        if self.generation % pop_config.epoch_size == 0 {
            self.mutation_rate *= pop_config.mutation_rate_scale_per_epoch;
            self.topology_mutation_rate *= pop_config.mutation_rate_scale_per_epoch;
            self.mutation_effect_sd *= pop_config.mutation_effect_scale_per_epoch;
        }

        if (self.generation % 10 == 0) & pop_config.visualize_best_agent {
            let agent_path = format!("{}{}_{:04}.dot", pop_config.project_directory, pop_config.project_name, self.generation);
            self.agents[best_agent_idx].nn.write_dot(&agent_path);
        }

        info!("Observing population fitness {} on generation {} with max of {}", self.population_fitness, self.generation, population_max);

    }

}

///## Unit Tests
///
///Unit tests are provided to validate the functionality of `Population` methods, including creation, fitness evaluation, truncation, SUS, offspring generation, and the overall evolution process with different configurations.
mod tests {
    use super::*;
    use crate::graph::NeuralNetwork;
    use crate::agent_wrapper::Agent;
    use crate::population::FitnessValueError;

    use crate::doctest::{GENOME_EXAMPLE, XOR_GENOME, XOR_GENOME_MINIMAL};
    use crate::setup_logger;

    struct XorEvaluation {
        pub fitness_begin: f32
    }

    impl XorEvaluation {
        pub fn new() -> Self {
            XorEvaluation {
                fitness_begin: 6.0
            }
        }

        pub fn evaluate_agent(&self, agent: &mut Agent) -> Result<(), FitnessValueError> {
            let mut fitness_evaluation = self.fitness_begin;
            //complexity penalty
            let complexity = agent.nn.node_identity_map.len() as f32;
            let complexity_penalty = 0.01 * complexity;

            for bit1 in 0..2 {
                for bit2 in 0..2 {
                    agent.fwd(vec![bit1 as f32, bit2 as f32]);
                    let network_output = agent.nn.fetch_network_output();

                    let xor_true = (bit1 > 0) ^ (bit2 > 0);
                    let xor_true_float: f32 = if xor_true {1.} else {0.};

                    fitness_evaluation -= (xor_true_float - network_output[0]).powf(2.);

                }
            }

            let fitness_value = if fitness_evaluation > complexity_penalty {
                fitness_evaluation - complexity_penalty }
            else {0.};

            if fitness_value < 0. {
                Err(FitnessValueError::NegativeFitnessError)
            } 
            else {
                agent.update_fitness(fitness_value);
                Ok(())
            }

        }
    }

    #[test]
    fn test_create_population() {
        let genome = GENOME_EXAMPLE.clone();
        let population_test = Population::new(genome, 125, 0.8, 0.1, 0.);
        assert_eq!(population_test.agents.len(), 125);
    }

    #[test]
    fn test_truncate_population() {
        let genome = GENOME_EXAMPLE.clone();
        let mut population_test = Population::new(genome, 10, 0.8, 0.1, 0.);
        population_test.agent_fitness = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        
        let sorted_fitness_indices = population_test.truncate_population();

        assert_eq!(sorted_fitness_indices, vec![9, 8, 7, 6, 5, 4, 3, 2]);
    }

    #[test]
    fn test_stochastic_universal_sampling() {
        let seed = [17; 32]; // Fixed seed for determinism
        let mut rng = StdRng::from_seed(seed);

        let genome = GENOME_EXAMPLE.clone();
        let mut population_test = Population::new(genome, 3, 0.8, 0.1, 0.);
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
        let population_test = Population::new(genome, 10, 0.8, 0.1, 0.);
        let parent_id_vector = vec![0, 1, 3, 5];

        let offspring_vec = population_test.generate_offspring(&mut rng, parent_id_vector);

        assert_eq!(offspring_vec.len(), 10);
    }

    #[test]
    fn test_evolve_xor_predefined_topology() {
        setup_logger();

        let genome = XOR_GENOME.clone();
        let mut population = Population::new(genome, 200, 0.1, 0.4, 0.01);

        let ef = XorEvaluation::new();
        

        while (population.generation < 400) & (population.population_fitness < 5.8) {
            for agent in population.agents.iter_mut() {
                ef.evaluate_agent(agent);
            }

            let config = PopulationConfig::new("XOR_Predefined_Test".to_string(), None, 50, 0.50, 0.50, false, Some(13));
            population.evolve_step(config);
        }

        assert!(population.population_fitness >= 5.2);
    }

    #[test]
    fn test_evolve_xor_minimal_topology() {
        setup_logger();

        let genome = XOR_GENOME_MINIMAL.clone();
        let mut population = Population::new(genome, 200, 0.2, 0.4, 0.4);

        let ef = XorEvaluation::new();

        while (population.generation < 1000) & (population.population_fitness < 5.8) {
            for agent in population.agents.iter_mut() {
                ef.evaluate_agent(agent);
            }

            let project_name = "XOR_Test".to_string();
            let project_directory = "agents/XORtest/".to_string();

            let config = PopulationConfig::new(project_name, Some(project_directory), 200, 0.50, 0.50, false, Some(17));
            population.evolve_step(config);
        }

        assert!(population.population_fitness >= 5.2);
    }

}
