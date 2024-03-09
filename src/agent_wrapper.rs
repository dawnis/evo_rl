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

pub struct AgentFactory {
    factory_type: String
}

impl AgentFactory {
    pub fn new<S: Into<String>>(s: S) -> Self {
        AgentFactory { factory_type: s.into() }
    }

    pub fn create(&self, agent: NeuralNetwork) -> Box<dyn NnWrapper> {
        if self.factory_type == "python" {
            Box::new(PythonAgent::new(agent)) as Box<dyn NnWrapper>
        }
        else {
            Box::new(NativeAgent::new(agent)) as Box<dyn NnWrapper>
        }
    }
}

pub trait NnWrapper: Send {
    fn fwd(&self, vector: Box<dyn NnInputVector>);
}

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

#[pyclass]
pub struct PythonAgent {
    nn: NeuralNetwork,
}

impl NnWrapper for PythonAgent {
    fn fwd(&self, vector: Box<dyn NnInputVector>) {
    }
}

#[pymethods]
impl PythonAgent {
    pub fn new(nn: NeuralNetwork) -> Self {
        PythonAgent { nn }
    }
}

