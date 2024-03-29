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

#[pyclass]
pub struct Agent {
    pub nn: Box<NeuralNetwork>,
    pub fitness: f32,
}

#[pymethods]
impl Agent {
    #[new]
    pub fn new(genome_base: EneCode) -> Self {
        let mut agent = NeuralNetwork::new(genome_base.clone());
        agent.initialize();
        // Random initialization of the population of all parameters
        agent.mutate(1., 10., 0.);
        Agent {
            nn: Box::new(agent),
            fitness: 0.
        }
    }

    pub fn fwd(&self, input: Vec<f32>) {
        self.nn.fwd(input);
    }

    pub fn update_fitness(&mut self, new_fitness: f32) {
        self.fitness = new_fitness;
    }
}

