//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::graph::NeuralNetwork;
use crate::population::{Population, PopulationConfig, FitnessEvaluation, FitnessValueError};
use crate::doctest::GENOME_EXAMPLE;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn evo_rl(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

/*
//TODO: Population::new

//#[pyclass]
/// Wrapper for Population
struct PopulationApi {
    population: Population,
    config: PyDict
}

#[pyclass]
struct PythonEvaluationFunction {
    py_evaluator: PyObject,
}

//#[pymethods]
impl PythonEvaluationFunction {
    #[new]
    fn new (py_evaluator: PyObject) {
        if py_evaluator.as_ref(py).cast_as::<PyCallable>().is_ok() {
            Ok( PythonEvaluationFunction { py_evaluator } )
        } else {
             Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Evaluation function is not allable",
            ))
        }
    }
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

    */


