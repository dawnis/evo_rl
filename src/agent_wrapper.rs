use crate::graph::NeuralNetwork;
use crate::enecode::EneCode;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};

pub trait NnInputVector: Send {
    fn into_vec_f32(&self) -> Vec<f32>;
}

impl NnInputVector for Vec<f32> {
    fn into_vec_f32(&self) -> Vec<f32> { 
        self.clone()
    }
}

pub struct AgentFactory {
    factory_type: String
}

impl AgentFactory {
    pub fn new<S: Into<String>>(s: S) -> Self {
        AgentFactory { factory_type: s.into() }
    }

    pub fn create(&self, genome_base: EneCode) -> Agent {
            Agent::new(genome_base, self.factory_type)
    }
}

pub trait NnWrapper: Send {
    fn fwd(&self, vector: Box<dyn NnInputVector>);
}

#[pyclass]
pub struct Agent {
    nn: Box<dyn NnWrapper>,
}

impl ToPyObject for Agent {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("nn", &self.nn);

        dict.into()
    }
}

#[pymethods]
impl Agent {
    #[new]
    pub fn new(genome_base: EneCode, environment: String) -> Self {
        let mut agent = NeuralNetwork::new(genome_base.clone());
        agent.initialize();
        // Random initialization of the population of all parameters
        agent.mutate(1., 10., 0.);
        let nn = if environment == "python" {
            Box::new(PythonWrapper::new(agent)) as Box<dyn NnWrapper>
        } else {
            Box::new(NativeWrapper::new(agent)) as Box<dyn NnWrapper>
        };

        Agent { nn }
    }
}

pub struct NativeWrapper {
    nn: NeuralNetwork,
}

impl NativeWrapper {
    pub fn new(nn: NeuralNetwork) -> Self {
        NativeWrapper { nn }
    }
}


impl NnWrapper for NativeWrapper {

    fn fwd(&self, vector: Box<dyn NnInputVector>) {
        self.nn.fwd(vector.into_vec_f32());
    }
}

pub struct PythonWrapper {
    nn: NeuralNetwork,
}

impl NnWrapper for PythonWrapper {
    fn fwd(&self, vector: Box<dyn NnInputVector>) {
    }
}

impl PythonWrapper {
    fn new(nn: NeuralNetwork) -> Self {
        PythonWrapper { nn }
    }
}

