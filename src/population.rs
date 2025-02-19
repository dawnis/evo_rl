//!This module focuses on implementing an evolutionary algorithm for neural network optimization. It uses Stochastic Universal Sampling (SUS) and Truncation for selection within a population of neural network agents.

use log::*;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand::Rng;
use thiserror::Error;

use reqwest::Url;

use std::io::Result as FileResult;
use std::path::PathBuf;
use std::sync::Arc;

use crate::agent_wrapper::*;
use crate::qdmanager::QDManager;
use crate::rng_box;
use crate::{enecode::EneCode, graph::NeuralNetwork};

/// `PopulationConfig` is a struct that configures `Population` for evolutionary selection.
///- **Purpose**: Configuration struct for setting hyperparameters of a population.
pub struct PopulationConfig {
    project_name: Arc<str>,
    project_directory: Arc<PathBuf>,
    api_endpoint: Option<Url>,
    rng_seed: Option<u8>,
    epoch_size: usize,
    mutation_rate_scale_per_epoch: f32,
    mutation_effect_scale_per_epoch: f32,
    visualize_best_agent: bool,
}

#[derive(Debug, Error)]
pub enum FitnessValueError {
    #[error("Negative fitness values are not allowed!")]
    NegativeFitnessError,
}

impl PopulationConfig {
    pub fn new(
        project_name: Arc<str>,
        home_directory: Option<Arc<PathBuf>>,
        api_endpoint: Option<Url>,
        epoch_size: usize,
        mutation_rate_scale_per_epoch: f32,
        mutation_effect_scale_per_epoch: f32,
        visualize_best_agent: bool,
        rng_seed: Option<u8>,
    ) -> Self {
        let project_directory: Arc<PathBuf> = match home_directory {
            Some(dir) => dir,
            None => PathBuf::from(".").into(),
        };

        PopulationConfig {
            project_name,
            project_directory,
            api_endpoint,
            rng_seed,
            epoch_size,
            mutation_rate_scale_per_epoch,
            mutation_effect_scale_per_epoch,
            visualize_best_agent,
        }
    }
}

///### `Population`
///- **Purpose**: Represents a population in the evolutionary algorithm.
pub struct Population {
    pub agents: Vec<Agent>,
    pub qdm: QDManager,
    pub size: usize,
    pub topology_mutation_rate: f32,
    pub mutation_rate: f32,
    pub mutation_effect_sd: f32,
    pub generation: usize,
    pub population_fitness: f32,
    survival_rate: f32,
}

impl Population {

    pub fn new(
        qdm: QDManager,
        population_size: usize,
        survival_rate: f32,
        mutation_rate: f32,
        topology_mutation_rate: f32,
    ) -> Self {

        let _ = qdm.init_library();

        let agent_vector: Vec<Agent> = qdm.gen_agent_vector(population_size);

        Population {
            agents: agent_vector,
            qdm,
            topology_mutation_rate,
            mutation_rate,
            mutation_effect_sd: 5.,
            size: population_size,
            generation: 0,
            population_fitness: 0.,
            survival_rate,
        }
    }

    ///### `selection`
    ///- **Purpose**: Selects a subset of agents from the population for reproduction.
    ///- **Parameters**:
    ///  - `n_select`: Number of agents to select.
    fn selection(&self, rng_seed: Option<u8>, n_select: usize) -> Vec<usize> {
        let truncated_population = self.truncate_population();
        self.stochastic_universal_sampling(rng_seed, truncated_population, n_select)
    }

    ///### `stochastic_universal_sampling`
    ///- **Purpose**: Implements SUS for efficient selection in evolutionary algorithms.
    fn stochastic_universal_sampling(
        &self,
        rng_seed: Option<u8>,
        sample: Vec<usize>,
        n_select: usize,
    ) -> Vec<usize> {
        let sample_fitness: Vec<f32> = sample.iter().map(|&idx| self.agents[idx].fitness).collect();
        let total_population_fitness: f32 = sample_fitness.iter().sum();
        let point_spacing = total_population_fitness / (n_select as f32);
        let u = Uniform::from(0_f32..point_spacing);

        let mut selection: Vec<usize> = Vec::new();

        //start the roulette pointer at a point between 0 and the first spacing

        let mut rng = rng_box(rng_seed);
        let mut roulette_pointer = u.sample(&mut rng);

        for _p in 0..n_select {
            let mut idx: usize = 0;
            while sample_fitness[0..idx].iter().sum::<f32>() < roulette_pointer {
                idx += 1;

                if idx == sample_fitness.len() {
                    break;
                }
            }
            selection.push(sample[idx - 1]);
            roulette_pointer += point_spacing;
        }

        selection
    }

    ///### `truncate_population`
    ///- **Purpose**: Truncates the population based on survival rate.
    ///- **Returns**: A vector of indices representing the surviving population.
    fn truncate_population(&self) -> Vec<usize> {
        let n_survival = (self.agents.len() as f32) * self.survival_rate;
        let agent_fitness: Vec<f32> = self.agents.iter().map(|a| a.fitness).collect();
        let mut fitness_pairing: Vec<_> = (0..self.agents.len()).zip(agent_fitness).collect();
        fitness_pairing.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let sorted_data: Vec<usize> = fitness_pairing.into_iter().map(|(a, _)| a).collect();

        sorted_data[..n_survival.round() as usize].to_vec()
    }

    ///### `generate_offspring`
    ///- **Purpose**: Generates offspring from selected parents.
    fn generate_offspring(&self, rng_seed: Option<u8>, parental_ids: Vec<usize>) -> Vec<Agent> {
        let mut offspring: Vec<Agent> = Vec::new();

        // Given selected parents, mate in pairs until the population size is fulfilled
        let num_parents = parental_ids.len();

        let mate_attempt_limit = self.size as f32 * 1.5;

        let mut n_mate_attempts = 0;

        let mut rng = rng_box(rng_seed);

        while offspring.len() < self.size {
            let a1 = rng.gen_range(0..num_parents);
            let partner_list: Vec<&usize> = parental_ids.iter().filter(|&&p| p != a1).collect();

            let a2 = rng.gen_range(0..partner_list.len());
            let parent_1 = parental_ids[a1];
            let parent_2 = *partner_list[a2];

            let offspring_ec = self.agents[parent_1]
                .nn
                .recombine_enecode(&mut rng, &self.agents[parent_2].nn);

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

    ///### `evolve_step`
    ///- **Purpose**: Runs a single round of evolution and increments one generation
    pub fn evolve_step(&mut self, pop_config: &PopulationConfig) {
        // Select same population size, but use SUS to select according to fitness
        let selection = self.selection(pop_config.rng_seed, self.size);

        let mut offspring = self.generate_offspring(pop_config.rng_seed, selection);

        for agent in offspring.iter_mut() {
            agent.mutate(
                self.mutation_rate,
                self.mutation_effect_sd,
                self.topology_mutation_rate,
            );
        }

        self.agents = offspring;

        self.generation += 1;

        if self.generation % pop_config.epoch_size == 0 {
            self.mutation_rate *= pop_config.mutation_rate_scale_per_epoch;
            self.topology_mutation_rate *= pop_config.mutation_rate_scale_per_epoch;
            self.mutation_effect_sd *= pop_config.mutation_effect_scale_per_epoch;
        }
    }

    pub fn update_population_fitness(&mut self) {
        let agent_fitness_vector: Vec<f32> = self.agents.iter().map(|a| a.fitness).collect();
        self.population_fitness = agent_fitness_vector.iter().sum::<f32>() / self.size as f32;
    }

    pub fn report(&self, pop_config: &PopulationConfig) {
        let agent_fitness_vector: Vec<f32> = self.agents.iter().map(|a| a.fitness).collect();
        let (best_agent_idx, population_max) = agent_fitness_vector
            .clone()
            .into_iter()
            .enumerate()
            .fold((0, std::f32::MIN), |(idx_max, val_max), (idx, val)| {
                if val > val_max {
                    (idx, val)
                } else {
                    (idx_max, val_max)
                }
            });

        if (self.generation % 10 == 0) & pop_config.visualize_best_agent {
            let dot_file = format!(
                "{}_{:04}.dot",
                pop_config.project_name.to_string(),
                self.generation
            );
            let agent_path = pop_config.project_directory.join(dot_file);
            info!("Writing agent dot with path {:?}", agent_path);
            self.agents[best_agent_idx].nn.write_dot(&agent_path);
        }

        info!(
            "Observing N={} population with fitness {} on generation {} with max of {} (agent {})",
            self.agents.len(),
            self.population_fitness,
            self.generation,
            population_max,
            best_agent_idx
        );
    }

    pub fn write_agent_genome(&self, idx: usize, file_path: PathBuf) -> FileResult<()> {
        self.agents[idx].write_genome(file_path)
    }


    pub fn post_agent_genome(&self, idx: usize) -> Result<(), reqwest::Error> {
        let agent_genome: EneCode = self.agents[idx].enecode();
        self.qdm.postg(&agent_genome)
    }

}

///## Unit Tests
///
///Unit tests are provided to validate the functionality of `Population` methods, including creation, fitness evaluation, truncation, SUS, offspring generation, and the overall evolution process with different configurations.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_wrapper::Agent;
    use crate::graph::NeuralNetwork;
    use crate::population::FitnessValueError;

    use crate::doctest::{GENOME_EXAMPLE, XOR_GENOME, XOR_GENOME_MINIMAL};
    use crate::setup_logger;

    struct XorEvaluation {
        pub fitness_begin: f32,
    }

    impl XorEvaluation {
        pub fn new() -> Self {
            XorEvaluation { fitness_begin: 6.0 }
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
                    let xor_true_float: f32 = if xor_true { 1. } else { 0. };

                    fitness_evaluation -= (xor_true_float - network_output[0]).powf(2.);
                }
            }

            let fitness_value = if fitness_evaluation > complexity_penalty {
                fitness_evaluation - complexity_penalty
            } else {
                0.
            };

            if fitness_value < 0. {
                Err(FitnessValueError::NegativeFitnessError)
            } else {
                debug!("Updating agent with fitness value {}", fitness_value);
                agent.update_fitness(fitness_value);
                Ok(())
            }
        }
    }

    #[test]
    fn test_create_population() {
        let genome = GENOME_EXAMPLE.clone();
        let qdm: QDManager = QDManager::new_from_genome(Arc::from("test"), None, genome);
        let population_test = Population::new(qdm, 125, 0.8, 0.1, 0.);
        assert_eq!(population_test.agents.len(), 125);
    }

    #[test]
    fn test_truncate_population() {
        let genome = GENOME_EXAMPLE.clone();
        let qdm: QDManager = QDManager::new_from_genome(Arc::from("test"), None, genome);
        let mut population_test = Population::new(qdm, 10, 0.8, 0.1, 0.);

        let agent_fitness_vector = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        for (agent, fitness) in population_test
            .agents
            .iter_mut()
            .zip(agent_fitness_vector.iter())
        {
            agent.fitness = *fitness;
        }

        let sorted_fitness_indices = population_test.truncate_population();

        assert_eq!(sorted_fitness_indices, vec![9, 8, 7, 6, 5, 4, 3, 2]);
    }

    #[test]
    fn test_stochastic_universal_sampling() {
        let seed = Some(17); // Fixed seed for determinism

        let genome = GENOME_EXAMPLE.clone();
        let qdm: QDManager = QDManager::new_from_genome(Arc::from("test"), None, genome);

        let mut population_test = Population::new(qdm, 3, 0.8, 0.1, 0.);
        let agent_fitness_vector = vec![5., 3., 2.];
        for (agent, fitness) in population_test
            .agents
            .iter_mut()
            .zip(agent_fitness_vector.iter())
        {
            agent.fitness = *fitness;
        }
        let sample: Vec<usize> = vec![0, 1, 2];

        let sus: Vec<usize> = population_test.stochastic_universal_sampling(seed, sample, 10);
        assert_eq!(vec![0, 0, 0, 0, 0, 1, 1, 1, 2, 2], sus);
    }

    #[test]
    fn test_generate_offspring() {
        let genome = GENOME_EXAMPLE.clone();
        let qdm: QDManager = QDManager::new_from_genome(Arc::from("test"), None, genome);
        let population_test = Population::new(qdm, 10, 0.8, 0.1, 0.);
        let parent_id_vector = vec![0, 1, 3, 5];

        let offspring_vec = population_test.generate_offspring(Some(17), parent_id_vector);

        assert_eq!(offspring_vec.len(), 10);
    }

    #[test]
    fn test_evolve_xor_predefined_topology() {
        setup_logger();

        let genome = XOR_GENOME.clone();
        let qdm: QDManager = QDManager::new_from_genome(Arc::from("test"), None, genome);
        let mut population = Population::new(qdm, 200, 0.1, 0.4, 0.01);

        let ef = XorEvaluation::new();

        let project_name: &str = "XOR_Predefined_Test";
        let config = PopulationConfig::new(
            Arc::from(project_name),
            None,
            None, 
            50,
            0.50,
            0.50,
            false,
            Some(13),
        );

        while (population.generation < 400) & (population.population_fitness < 5.8) {
            for agent in population.agents.iter_mut() {
                ef.evaluate_agent(agent);
            }

            population.update_population_fitness();
            population.report(&config);
            population.evolve_step(&config);
        }

        assert!(population.population_fitness >= 5.2);
    }

    #[test]
    fn test_evolve_xor_minimal_topology() {
        setup_logger();

        let genome = XOR_GENOME_MINIMAL.clone();
        let qdm: QDManager = QDManager::new_from_genome(Arc::from("test"), None, genome);
        let mut population = Population::new(qdm, 200, 0.2, 0.4, 0.4);

        let ef = XorEvaluation::new();

        let project_name: &str = "XOR_Test";
        let project_directory: PathBuf = PathBuf::from("agents/XORtest");

        let config = PopulationConfig::new(
            Arc::from(project_name),
            Some(Arc::from(project_directory)),
            None,
            200,
            0.50,
            0.50,
            false,
            Some(237),
        );

        while (population.generation < 1000) & (population.population_fitness < 5.8) {
            for agent in population.agents.iter_mut() {
                let _ = ef.evaluate_agent(agent);
            }

            population.update_population_fitness();
            population.report(&config);
            population.evolve_step(&config);
        }

        assert!(population.population_fitness >= 5.2)
    }
}
