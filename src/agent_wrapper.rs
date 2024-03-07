use crate::graph::NeuralNetwork;
use pyo3::prelude::*;
use pyo3::types::PyList;

pub trait NnInputVector: Send {
    fn into_vec_f32(&self) -> Vec<f32>;
}

impl NnInputVector for Vec<f32> {
    fn into_vec_f32(&self) -> Vec<f32> { 
        self.clone()
    }
}

pub trait NnWrapperFactory {
    fn create(&self, agent: NeuralNetwork) -> Box<dyn NnWrapper>;
}

pub struct AgentFactory {
    factory_type: String
}

impl AgentFactory {
    pub fn new(factory_type: str) -> Self {
        AgentFactory { factory_type }
    }
}

impl NnWrapperFactory for AgentFactory {
    fn create(&self, agent: NeuralNetwork) -> Box<dyn NnWrapper> {
        Box::new(NativeAgent::new(agent)) as Box<dyn NnWrapper>
    }
}

pub trait NnWrapper: Send {
    fn fwd(&self, vector: Box<dyn NnInputVector>);
}

//TODO: Wrapper for native rust
pub struct NativeAgent {
    nn: NeuralNetwork,
}

impl NativeAgent {
    pub fn new(nn: NeuralNetwork) -> Self {
        NativeAgent { nn }
    }
}


impl NnWrapper for NativeAgent {

    fn fwd(&self, vector: Box<dyn NnInputVector>) {
        self.nn.fwd(vector.into_vec_f32());
    }
}

//TODO: this is specific for the python API
#[pyclass]
pub struct AgentWrapper {
    nn: NeuralNetwork,
}

impl NnWrapper for AgentWrapper {
    fn fwd(&self, vector: Box<dyn NnInputVector>) {
    }
}
#[pymethods]
impl AgentWrapper {
}

