//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use log::*;
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyDict, PyList, IntoPyDict, PyTuple, PyFloat};
use pyo3::Py;
use std::collections::HashMap;
use std::cell::Cell;
use crate::graph::NeuralNetwork;
use crate::agent_wrapper::*;
use crate::population::{Population, PopulationConfig, FitnessEvaluation, FitnessValueError};
use crate::doctest::{GENOME_EXAMPLE, XOR_GENOME, XOR_GENOME_MINIMAL};


#[pyfunction]
fn log_something() {
    // This will use the logger installed in `my_module` to send the `info`
    // message to the Python logging facilities.
    info!("This is a test of pyo3-logging.");
}

/// A Python module for evo_rl implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn evo_rl(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<PopulationApi>()?;
    m.add_class::<PyFitnessEvaluator>()?;
    let _ = m.add_function(wrap_pyfunction!(log_something, m)?);
    Ok(())
}

//TODO: more verbose output in Python
//TODO: genome specification from python
//TODO: nework visualization and exploration


//TODO: a layer of abstraction is required over nn to allow it to be used in different environments
#[pyclass(name = "FitnessEvaluator")]
pub struct PyFitnessEvaluator {
    pub lambda: Py<PyAny>
}

impl FitnessEvaluation for PyFitnessEvaluator{
    fn fitness(&self, agent: Box<dyn NnWrapper>)-> Result<f32, FitnessValueError> {

        let mut agent_mut = agent.clone();

        let fitness_value_py_result = Python::with_gil(|py| -> PyResult<f32> {
            let args = PyTuple::empty(py);
        
            let kwargs = PyDict::new(py);
            let _ = kwargs.set_item("agent", agent_mut);

            let lambda_call = self.lambda.call(py, args, Some(kwargs));

            match lambda_call {
                Ok(x) => Ok(x.extract::<f32>(py)?),
                Err(e) => panic!("{}", e)
            }
        });


        match fitness_value_py_result {
            Ok(fitness) => Ok(fitness),
            Err(e) => panic!("{}", e)
        }
    }
}

impl<'source> FromPyObject<'source> for PyFitnessEvaluator {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let py = obj.py();

        let py_func: Py<PyAny> = obj.getattr("__call__")?.into();


/*
        let func = obj.downcast::<PyFunction>()?;

        let py_function = func.get_item("lambda")
            .or_else(|py| Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected a lambda field")))?
            .to_object(py); 
            */

        Ok(
            PyFitnessEvaluator { lambda: py_func }
        )
    }
}

#[pymethods]
impl PyFitnessEvaluator {
    #[new]
    fn __new__(lambda: Py<PyAny>) -> Self {
        info!("Building a Fitness Evaluator in Python");
        PyFitnessEvaluator { lambda }
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__ (
        &self,
        py: Python<'_>,
        args: &PyTuple,
        kwargs: Option<&PyDict>,
        ) -> PyResult<Py<PyFloat>> {

        let call_result = self.lambda.call(py, args, kwargs)?;
        let call_result_float = call_result.extract::<Py<PyFloat>>(py)?;

        Ok(call_result_float)
    }

    /*
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
    */

}

#[pyclass]
/// Wrapper for Population
struct PopulationApi {
    population: Population,
    evaluator: Py<PyFitnessEvaluator>,
    config: Py<PyDict>
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

    pub fn evolve(&mut self) {
        let project_name = "XOR_Test".to_string();
        let project_directory = "agents/XORtest/".to_string();


        let py_context = Python::with_gil(|py| -> PyResult<PyFitnessEvaluator> {
            let py_evalutor = self.evaluator.extract(py)?;
            Ok(py_evalutor)
        });

        match py_context {
            Ok(context) => {
                let config = PopulationConfig::new(project_name, Some(project_directory), context, 200, 0.50, 0.50, false, Some(17));
                self.population.evolve(config, 1000, 5.8);
            },

            Err(e) => error!("Unable to unpack Python context for population with error {}", e)
        };

    }


}

