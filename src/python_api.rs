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
use std::path::PathBuf;


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
    evolve_config: Box<PopulationConfig>,
    pop_config: Py<PyDict>
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

        let evolve_config = Python::with_gil(|py| -> PyResult<PopulationConfig> {

            let config: &PyDict = pyconfig.as_ref(py);

            let project_name: String = match config.get_item("project_name")? {
                Some(x) => x.extract()?,
                None => panic!("missing project name")
            };

            let project_directory: String = match config.get_item("project_directory")? {
                Some(x) => x.extract()?,
                None => panic!("missing project directory")
            };

            let project_path: Arc<PathBuf> = Arc::from(PathBuf::from(project_directory));

            Ok(PopulationConfig::new(Arc::from(project_name), Some(project_path), 200, 0.50, 0.50, true, None))

        })?;


        Ok(PopulationApi {
            population: Box::new(population),
            evolve_config: Box::new(evolve_config),
            pop_config: pyconfig
        })
    }

    pub fn evolve_step(&mut self) {
        self.population.evolve_step(&self.evolve_config);
    }

    pub fn update_population_fitness(&mut self) {
        self.population.update_population_fitness();
    }

    pub fn report(&self) {
        self.population.report(&self.evolve_config);
    }

    pub fn agent_fwd(&mut self, idx: usize, input: Py<PyList>) {
        let py_vec = Python::with_gil(|py| -> Result<Vec<f32>, PyErr> {
            let input_vec = input.as_ref(py);
            input_vec.iter()
                .map(|p| p.extract::<f32>())
                .collect()
            });

        match py_vec {
            Ok(v) => self.population.agents[idx].fwd(v),
            err => error!("PyError: {:?}", err)
        }

    }

    pub fn agent_out(&self, idx: usize) -> PyResult<Vec<f32>> {
        Ok(self.population.agents[idx].output())
    }

    pub fn agent_complexity(&self, idx: usize) -> PyResult<usize> {
        Ok(self.population.agents[idx].nn.node_identity_map.len())
    }

    pub fn set_agent_fitness(&mut self, idx: usize, value: f32) {
        self.population.agents[idx].fitness = value;
    }

    #[getter(generation)]
    fn generation(&self) -> PyResult<usize> {
        Ok(self.population.generation)
    }

    #[getter(fitness)]
    fn fitness(&self) -> PyResult<f32> {
        Ok(self.population.population_fitness)
    }


}

