//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyDict, PyList};
use pyo3::Py;
use crate::graph::NeuralNetwork;
use crate::population::{Population, PopulationConfig, FitnessEvaluation, FitnessValueError};
use crate::doctest::{GENOME_EXAMPLE, XOR_GENOME, XOR_GENOME_MINIMAL};


//TODO: Try to expose a Python API for Neural Network
// This api must allow a configuration of a stopping condition for forward
// and evaluation conditions for the NN. This will allow us to pass this into evolve
// as the proper context
#[pyclass]
struct PyNetwork {
   // network: NeuralNetwork
}

#[pymethods]
impl PyNetwork {

    #[staticmethod]
    pub fn new() -> Self {
        PyNetwork {}
    }

    pub fn agent_forward(&mut self, network_arguments: Py<PyList>) {

        let py_vec = Python::with_gil(|py| -> Result<Vec<f32>, PyErr> {
            let input_vec = network_arguments.as_ref(py);

            input_vec.iter()
                .map(|p| p.extract::<f32>())
                .collect()

        });

    }
}

struct PyLambda<'a> {
        pub fitness_begin: f32,
        pub lambda: &'a PyFunction,
    }

impl<'a> PyLambda<'a> {
    pub fn new(lambda: &PyFunction) -> Self {
        PyLambda {
            fitness_begin: 6.0,
            lambda

        }
    }

}

    impl<'a> FitnessEvaluation for PyLambda<'a> {
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


//TODO: more verbose output in Python
//TODO: be able to pass in evaluation function from Python
//TODO: genome specification from python
//TODO: nework visualization and exploration

#[pymethods]
impl PopulationApi {
    #[new]
    pub fn new(pyconfig: Py<PyDict>) -> PyResult<Self> {
        let genome = XOR_GENOME_MINIMAL.clone();


        let population = Python::with_gil(|py| -> PyResult<Population> {

            let config: &PyDict = pyconfig.as_ref(py);

            let population_size: usize = match config.get_item("population_size")? {
                Some(x) => x.extract()?,
                None => panic!("missing population size parameter")
            };

            let survival_rate: f32  = match config.get_item("survival_rate")? {
                Some(x) => x.extract()?,
                None => panic!("missing population survival rate parameter")
            };

            let mutation_rate: f32  = match config.get_item("mutation_rate")? {
                Some(x) => x.extract()?,
                None => panic!("missing population mutation rate parameter")
            };

            let topology_mutation_rate: f32  = match config.get_item("topology_mutation_rate")? {
                Some(x) => x.extract()?,
                None => panic!("missing population topology rate parameter")
            };


            Ok(Population::new(genome, population_size, survival_rate, mutation_rate, topology_mutation_rate))

        })?;

        Ok(PopulationApi {
            population,
            config: pyconfig
        })
    }

    //TODO: evaluate must involve communication between population and python
    // 
    fn evaluate_agent(&mut self, context: &PyFunction) -> f32 {

        1.
    }

    pub fn evolve(&mut self, context: &PyFunction) {
        let project_name = "XOR_Test".to_string();
        let project_directory = "agents/XORtest/".to_string();
        
        let ef = PyLambda::new();

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
