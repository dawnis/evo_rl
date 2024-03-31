//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use log::*;
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyDict, PyList, IntoPyDict, PyTuple, PyFloat};
use pyo3::Py;
use std::collections::HashMap;
use std::cell::Cell;
use std::sync::Arc;
use crate::graph::NeuralNetwork;
use crate::agent_wrapper::*;
use crate::population::{Population, PopulationConfig, FitnessValueError};
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
    let _ = m.add_function(wrap_pyfunction!(log_something, m)?);
    Ok(())
}

#[pyclass]
/// Wrapper for Population
struct PopulationApi {
    population: Box<Population>,
    config: Py<PyDict>
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

        let test_name = String::from("python_test");
        //let project_name = "XOR_Test".to_string();
        //let project_directory = "agents/XORtest/".to_string();



        Ok(PopulationApi {
            population: Box::new(population),
            config: pyconfig
        })
    }

    pub fn evolve_step(&mut self) {
        let test_name: &str = "test_name";
        let step_configuration = PopulationConfig::new(Arc::from(test_name), None, 200, 0.50, 0.50, false, Some(17));
        self.population.evolve_step(&step_configuration);
    }


}

