//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use log::*;
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyDict, PyList, IntoPyDict, PyTuple, PyFloat};
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
    m.add_class::<PyFitnessEvaluator>()?;
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

impl FitnessEvaluation for PyFitnessEvaluator{
    fn fitness(&self, agent: &mut NeuralNetwork) -> Result<f32, FitnessValueError> {
        //TODO: define the signature of the lambda function.
        //def evaluate_fitness(x: n.agent_forward):
        //    runs agent_forward n times and returns fitness value

        let fitness_value_py_result = Python::with_gil(|py| -> PyResult<f32> {
            let args = PyTuple::empty_bound(py);
        
            // call object with PyDict
            let kwargs = [("agent", 1)].into_py_dict(py);

            let lambda_call = self.lambda.call_bound(py, args, Some(&kwargs.as_borrowed()));

            match lambda_call {
                Ok(x) => Ok(x.extract::<f32>(py)?),
                Err(e) => panic!("Error {}", e)
            }
        });


        match fitness_value_py_result {
            Ok(fitness) => Ok(fitness),
            Err(e) => panic!("Error {}", e)
        }
    }
}

impl<'source> FromPyObject<'source> for PyFitnessEvaluator {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let py = obj.py();

        let dict = obj.downcast::<PyDict>()?;

        let py_function = dict.get_item("lambda")
            .or_else(|py| Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected a lambda field")))?
            .to_object(py); 

        Ok(
            PyFitnessEvaluator { lambda: py_function }
        )
    }
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
        ) -> PyResult<Py<PyFloat>> {

        let call_result = self.lambda.call_bound(py, args, kwargs.as_ref())?;
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
fn test() -> PyResult<()> {
    let key1 = "key1";
    let val1 = 1;
    let key2 = "key2";
    let val2 = 2;

    Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            "def example(*args, **kwargs):
                if args != ():
                    print('called with args', args)
                if kwargs != {}:
                    print('called with kwargs', kwargs)
                if args == () and kwargs == {}:
                    print('called with no arguments')",
            "",
            "",
        )?
        .getattr("example")?
        .into();

        // call object with PyDict
        let kwargs = [(key1, val1)].into_py_dict(py);
        fun.call_bound(py, (), Some(&kwargs.as_borrowed()))?;

        // pass arguments as Vec
        let kwargs = vec![(key1, val1), (key2, val2)];
        fun.call_bound(py, (), Some(&kwargs.into_py_dict_bound(py)))?;

        // pass arguments as HashMap
        let mut kwargs = HashMap::<&str, i32>::new();
        kwargs.insert(key1, 1);
        fun.call_bound(py, (), Some(&kwargs.into_py_dict_bound(py)))?;

        Ok(())
    })
}
