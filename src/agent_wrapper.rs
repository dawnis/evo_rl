//! A wrapper class for neural network that creates an interface to other environments such as
//! Python
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
        let agent = NeuralNetwork::new(genome_base.clone());

        Agent {
            nn: Box::new(agent),
            fitness: 0.
        }
    }

    pub fn fwd(&mut self, input: Vec<f32>) {
        self.nn.fwd(input);
    }

    pub fn output(&self) -> Vec<f32> {
        self.nn.fetch_network_output()
    }

    pub fn mutate(&mut self, mutation_rate: f32, mutation_sd: f32, topology_mutation_rate: f32) {
        self.nn.mutate(mutation_rate, mutation_sd, topology_mutation_rate);
    }

    pub fn update_fitness(&mut self, new_fitness: f32) {
        self.fitness = new_fitness;
    }
}

