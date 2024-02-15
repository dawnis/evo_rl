//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use log::*;
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyDict, PyList};
use pyo3::Py;
use crate::graph::NeuralNetwork;
use crate::population::{Population, PopulationConfig, FitnessEvaluation, FitnessValueError};
use crate::doctest::{GENOME_EXAMPLE, XOR_GENOME, XOR_GENOME_MINIMAL};


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
//TODO: The evaluation of the network should be external to evo_rl library in general. 
//In order to accomplish this, we should have a Python enabled FitnessEvaluation trait that works
//with PyNetworkApi 


struct PyNetworkApi<'a> {
    pub lambda: &'a PyFunction,
}

impl<'a> PyNetworkApi<'a> {

    pub fn new(lambda: &PyFunction) -> Self {
        PyNetworkApi {lambda}
    }

    pub fn agent_forward(&self, nn: &mut NeuralNetwork, network_arguments: Py<PyList>) {

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
}

impl<'a> FitnessEvaluation for PyNetworkApi<'a> {
    fn fitness(&self, agent: &mut NeuralNetwork) -> Result<f32, FitnessValueError> {
        //TODO: define the signature of the lambda function.
        //def evaluate_fitness(x: n.agent_forward):
        //    runs agent_forward n times and returns fitness value
        

        //TODO: run an agent through the lambda function until a stop signal is sent. 
        //call lambda on self.agent_forward, which returns the fitness
        
        let fitness_eval = Python::with_gil(|py| -> f32 {
            let kwargs = [("agent_forward", self.agent_forward(agent, vec![0.]))].into_py_dict(py);

            let py_evaluation = self.lambda.call(py, (), Some(kwargs))?;

            match py_evaluation {
                Some(x) => x.extract().as_ref(),
                _ => PyErr
            }
        });

        //TODO: get the fitness value from the lambda function
        //return  the fitness value
        Ok(fitness_eval)
    }
}


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
