//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use pyo3::prelude::*;
use crate::population::Population;
use crate::doctest::GENOME_EXAMPLE;

//TODO: Population::new

#[pyclass]
/// Wrapper for Population
struct PopulationApi {
    population: Population,
}

#[pymethods]
impl PopulationApi {
    #[new]
    fn new(population_size: usize, survival_rate: f32, mutation_rate: f32, topology_mutation_rate: f32) -> Self {
        let genome = GENOME_EXAMPLE.clone();
        let mut population = Population::new(genome, population_size, survival_rate, mutation_rate, topology_mutation_rate);
        PopulationApi { population }
    }

    fn evolve(&mut self, ...)
}

//TODO: population config
//TODO: Fitness eval loop


