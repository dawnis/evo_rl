//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::graph::NeuralNetwork;
use crate::population::{Population, PopulationConfig, FitnessEvaluation, FitnessValueError};
use crate::doctest::GENOME_EXAMPLE;

//TODO: Population::new

#[pyclass]
/// Wrapper for Population
struct PopulationApi {
    population: Population,
    config: PyDict
}

#[pyclass]
struct PythonEvaluationFunction {
    pyevaluator: PyObject,
    py: Python,
}

impl FitnessEvaluation for PythonEvaluationFunction {
    fn fitness(&self, agent: &mut NeuralNetwork) -> Result<f32, FitnessValueError> {
        let eval = self.pyevaluator.as_ref(self.py);
        eval.call0()
    }
}

#[pymethods]
impl PopulationApi {
    #[new]
    pub fn new(config: PyDict, population_size: usize, survival_rate: f32, mutation_rate: f32, topology_mutation_rate: f32) -> Self {
        let genome = GENOME_EXAMPLE.clone();
        let mut population = Population::new(genome, population_size, survival_rate, mutation_rate, topology_mutation_rate);
        PopulationApi { 
            population,
            config
        }
    }

    fn config_from_python_dict(py_dict: &PyDict) -> PopulationConfig {
        let project_name = "XOR_Test".to_string();
        let project_directory = "agents/XORtest/".to_string();
        
        struct XorEvaluation {
            pub fitness_begin: f32
        }

        impl XorEvaluation {
            pub fn new() -> Self {
                XorEvaluation {
                    fitness_begin: 6.0
                }
            }

        }
        let ef = XorEvaluation::new();
        let config = PopulationConfig::new(project_name, Some(project_directory), ef, 200, 0.50, 0.50, false, Some(17));
        config
    }

}

//TODO: population config
//TODO: Fitness eval loop


