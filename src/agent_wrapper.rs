//! A wrapper class for neural network that creates an interface to other environments such as
//! Python
use crate::graph::NeuralNetwork;
use crate::enecode::EneCode;


use pyo3::prelude::*;

use std::path::PathBuf;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::io::Result as FileResult;
use std::error::Error;

pub trait NnInputVector: Send {
    fn into_vec_f64(&self) -> Vec<f64>;
}

impl NnInputVector for Vec<f64> {
    fn into_vec_f64(&self) -> Vec<f64> { 
        self.clone()
    }
}

#[pyclass]
pub struct Agent {
    pub nn: Box<NeuralNetwork>,
    pub fitness: f64,
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

    pub fn fwd(&mut self, input: Vec<f64>) {
        self.nn.fwd(input);
    }

    pub fn output(&self) -> Vec<f64> {
        self.nn.fetch_network_output()
    }

    pub fn mutate(&mut self, mutation_rate: f64, mutation_sd: f64, topology_mutation_rate: f64) {
        self.nn.mutate(mutation_rate, mutation_sd, topology_mutation_rate);
    }

    pub fn update_fitness(&mut self, new_fitness: f64) {
        self.fitness = new_fitness;
    }

    pub fn enecode(&self) -> EneCode {
        let nn_deref = *self.nn.clone();
        EneCode::from(&nn_deref)
    }

    pub fn write_genome(&self, file_path: PathBuf) -> FileResult<()> {
        let serialized_genome = self.nn.serialize_genome();
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(serialized_genome.as_bytes())?;
        writer.flush()?;
        Ok(())
    }

}

