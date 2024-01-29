//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;
use crate::graph::NeuralNetwork;
use crate::population::{Population, PopulationConfig, FitnessEvaluation, FitnessValueError};
use crate::doctest::{GENOME_EXAMPLE, XOR_GENOME, XOR_GENOME_MINIMAL};

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

    impl FitnessEvaluation for XorEvaluation {
        fn fitness(&self, agent: &mut NeuralNetwork) -> Result<f32, FitnessValueError> {
            let mut fitness_evaluation = self.fitness_begin;
            //complexity penalty
            let complexity = agent.node_identity_map.len() as f32;
            let complexity_penalty = 0.01 * complexity;

            for bit1 in 0..2 {
                for bit2 in 0..2 {
                    agent.fwd(vec![bit1 as f32, bit2 as f32]);
                    let network_output = agent.fetch_network_output();

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
                Ok(fitness_value) 
            }

        }
    }

/// A Python module for evo_rl implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn evo_rl(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PopulationApi>()?;
    Ok(())
}

#[pyclass]
/// Wrapper for Population
struct PopulationApi {
    population: Population,
    config: Py<PyDict>
}


#[pymethods]
impl PopulationApi {
    #[new]
    pub fn new(pyconfig: Py<PyDict>) -> PyResult<Self> {
        let genome = XOR_GENOME_MINIMAL.clone();


        Python::with_gil(|py| -> PyResult<()> {
            let config: &PyDict = pyconfig.as_ref(py);
            if let Some(size) = config.get_item("population_size")? {
                let population_size: usize = size.extract()?;
            }

            if let Some(rate) = config.get_item("survival_rate")? {
                let survival_rate: f32 = rate.extract()?;
            }

            if let Some(rate) = config.get_item("mutation_rate")? {
                let mutation_rate: f32 = rate.extract()?;
            }

            if let Some(rate) = config.get_item("topology_mutation_rate")? {
                let topology_mutation_rate: f32 = rate.extract()?;
            }

            Ok(())
        });

        let population = Population::new(genome, population_size, survival_rate, mutation_rate, topology_mutation_rate);
        Ok(PopulationApi {
            population,
            config: pyconfig
        })
    }

    //TODO: run xor_minimal_test as is from Python
    pub fn evolve(&mut self) {
        let project_name = "XOR_Test".to_string();
        let project_directory = "agents/XORtest/".to_string();
        
        let ef = XorEvaluation::new();

        let config = PopulationConfig::new(project_name, Some(project_directory), ef, 200, 0.50, 0.50, false, Some(17));

        self.population.evolve(config, 1000, 5.8);
    }


}


/*
#[pyclass]
struct PythonEvaluationFunction {
    py_evaluator: PyObject,
}

#[pymethods]
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
*/
