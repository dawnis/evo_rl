use crate::graph::NeuralNetwork;
use pyo3::prelude::*;
use pyo3::types::PyList;

pub trait NnInputVector {
    fn into_vec_f32(&self) -> Vec<f32>;
}
pub trait NnWrapper {
    fn fwd(&self, vector: Py<PyList>);
}

pub struct AgentWrapper {
}

