//! This module implements a Python API for Evo RL to allow training and running Population in
//! Python
use log::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Py;
use crate::enecode::EneCode;
use crate::agent_wrapper::Agent;
use std::path::PathBuf;
use pyo3::exceptions::PyRuntimeError;

#[pyclass]
pub struct AgentApi {
    agent: Agent,
    topology_mutation_rate: f32,
    synaptic_mutation_rate: f32,
    mutation_effect: f32
}

#[pymethods]
impl AgentApi {
    #[new]
    pub fn new(pyconfig: Py<PyDict>, checkpoint: Option<PathBuf>) -> PyResult<Self> {

            Python::with_gil(|py| -> PyResult<AgentApi> {

                let config: &PyDict = pyconfig.as_ref(py);

                let synaptic_mutation_rate: f32  = match config.get_item("synaptic_mutation_rate")? {
                    Some(x) => x.extract()?,
                    None => panic!("missing synaptic mutation rate parameter")
                };

                let topology_mutation_rate: f32  = match config.get_item("topology_mutation_rate")? {
                    Some(x) => x.extract()?,
                    None => panic!("missing topology mutation parameter")
                };

                let mutation_effect: f32  = match config.get_item("mutation_effect")? {
                    Some(x) => x.extract()?,
                    None => panic!("missing mutation effect parameter")
                };

                let genome: EneCode = match checkpoint {
                    Some(chkpt) => match EneCode::try_from(&chkpt) {
                        Ok(enecode) => enecode,
                        Err(err) => panic!("{}", err)
                    },
                    None => panic!("Construction of single agents only supported by checkpoint at this time.")
                };

                Ok(AgentApi {
                    agent: Agent::new(genome),
                    synaptic_mutation_rate,
                    topology_mutation_rate,
                    mutation_effect
                })

            })

    }

    /// Evaluates an agent's otuput given an input vector.
    pub fn fwd(&mut self, input: Py<PyList>) {
        let py_vec = Python::with_gil(|py| -> Result<Vec<f32>, PyErr> {
            let input_vec = input.as_ref(py);
            input_vec.iter()
                .map(|p| p.extract::<f32>())
                .collect()
            });

        match py_vec {
            Ok(v) => self.agent.fwd(v),
            err => error!("PyError: {:?}", err)
        }
    }


    /// Gets the agent's current output value. 
    pub fn output(&self) -> PyResult<Vec<f32>> {
        Ok(self.agent.output())
    }
}

