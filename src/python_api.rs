//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use log::*;
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyDict, PyList, IntoPyDict, PyTuple};
use pyo3::Py;
use std::collections::HashMap;
use std::cell::Cell;
use crate::graph::NeuralNetwork;
use crate::population::{Population, PopulationConfig, FitnessEvaluation, FitnessValueError};
use crate::doctest::{GENOME_EXAMPLE, XOR_GENOME, XOR_GENOME_MINIMAL};


/// A Python module for evo_rl implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn evo_rl(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PopulationApi>()?;
    m.add_class::<PyFitnessEvaluator>();
    Ok(())
}

//TODO: more verbose output in Python
//TODO: be able to pass in evaluation function from Python
//TODO: genome specification from python
//TODO: nework visualization and exploration
//TODO: The evaluation of the network should be external to evo_rl library in general. 
//In order to accomplish this, we should have a Python enabled FitnessEvaluation trait that works
//with PyNetworkApi 

//see https://pyo3.rs/main/class/call

//TODO: Goal is to pass in a callable class which wraps the evaluation function. 
#[pyclass(name = "FitnessEvaluator")]
pub struct PyFitnessEvaluator {
    lambda: Py<PyAny>
}

#[pymethods]
impl PyFitnessEvaluator {
    fn new(lambda: Py<PyAny>) -> Self {
        PyFitnessEvaluator { lambda }
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__ (
        &self,
        py: Python<'_>,
        args: &PyTuple,
        kwargs: Option<Bound<'_, PyDict>>,
        ) -> PyResult<Py<PyAny>> {

        let call_result = self.lambda.call_bound(py, args, kwargs.as_ref())?;

        Ok(call_result)
    }
}

#[pyclass]
/// Wrapper for Population
struct PopulationApi {
    population: Population,
    evaluator: Py<PyFitnessEvaluator>,
    config: Py<PyDict>
}

impl FitnessEvaluation for PopulationApi {
    fn fitness(&self, agent: &mut NeuralNetwork) -> Result<f32, FitnessValueError> {
        //TODO: define the signature of the lambda function.
        //def evaluate_fitness(x: n.agent_forward):
        //    runs agent_forward n times and returns fitness value

        let fitness_eval = 1.;

        //TODO: get the fitness value from the lambda function
        //return  the fitness value
        Ok(fitness_eval)
    }
}


#[pymethods]
impl PopulationApi {
    #[new]
    pub fn new(pyconfig: Py<PyDict>, context: Py<PyFitnessEvaluator>) -> PyResult<Self> {
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
            evaluator: context,
            config: pyconfig
        })
    }

    fn agent_forward(&self, nn: &mut NeuralNetwork, network_arguments: Py<PyList>) {

    let py_vec = Python::with_gil(|py| -> Result<Vec<f32>, PyErr> {
        let input_vec = network_arguments.as_ref(py);

        input_vec.iter()
            .map(|p| p.extract::<f32>())
            .collect()

        });

        match py_vec {
            Ok(v) => nn.fwd(v),
            err => error!("PyError: {:?}", err)
        }

    }

    pub fn evolve(&mut self, context: &PyFunction) {
        let project_name = "XOR_Test".to_string();
        let project_directory = "agents/XORtest/".to_string();
        
        let ef = PyLambda::new();

        let config = PopulationConfig::new(project_name, Some(project_directory), ef, 200, 0.50, 0.50, false, Some(17));

        self.population.evolve(config, 1000, 5.8);
    }


}


