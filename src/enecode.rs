use std::collections::HashMap;
use crate::neuron::Nn;

//Evolvable Topological Encoding
// TODO:
// Chromosomal Modules:
//  1. topology and actives/not-active
//  2. integration constants, thresholds, non-linearity parameters
//  3. meta-learning such as synaptic plasticity rules
//

pub trait Gene {
}

pub struct EneCode {
    pub genome: Vec<Chromosome>
}

pub struct Chromosome {
    pub encoding: Vec<Box<dyn Gene>>
}


pub struct ToplogyGene {
    pub identity: Nn,
    pub outputs: Vec<Nn>,
    pub active: bool,
    pub innovation_number: usize,
}

impl Gene for ToplogyGene {
}


pub struct NeuronalPropertiesGene {
    pub identity: Nn,
    pub innovation_number: usize,
    pub tau: f32,
    pub homeostatic_force: f32,
    pub alpha: f32
}

impl Gene for NeuronalPropertiesGene {
}


pub struct MetalearningGene {
    pub identity: Nn,
    pub innovation_number: usize,
    pub learning_rate: f32,
    pub learning_threshold: f32
}
