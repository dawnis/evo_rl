use std::collections::HashMap;
use crate::neuron::Nn;

//Evolvable Topological Encoding
// TODO:
// Chromosomal Modules:
//  1. topology and actives/not-active
//  2. integration constants, thresholds, non-linearity parameters
//  3. meta-learning such as synaptic plasticity rules
//

//Overall Genome
#[derive(Debug, Clone)]
pub struct EneCode {
    pub neuron_id: Vec<String>, //equivalent to innovation_number in TopologyGene
    pub topology: Vec<TopologyGene>,
    pub neuronal_props: NeuronalPropertiesGene,
    pub meta_learning: MetaLearningGene,
}

impl EneCode {
    pub fn topology_gene(&self, neuron_id: &String) -> &TopologyGene {
        let gene = self.topology.iter()
            .find(|&g| g.innovation_number == *neuron_id)
            .expect("Innovation Number Not Found in Topology Genome!");

        gene
    }
}

//Genetic Makeup of an Individual Neuron
pub struct NeuronalEneCode {
    pub neuron_id: String,
    pub topology: TopologyGene,
    pub properties: NeuronalPropertiesGene,
    pub meta: MetaLearningGene,
}

impl NeuronalEneCode {
    pub fn new_from_enecode(neuron_id: String, genome: &EneCode) -> Self {
        let topology = genome.topology_gene(&neuron_id).clone();
        NeuronalEneCode {
            neuron_id,
            topology,
            properties: genome.neuronal_props.clone(),
            meta: genome.meta_learning.clone(), 
        }
    }
}

#[derive(Debug, Clone)]
pub enum NeuronType {
    In,
    Out,
    Hidden,
}

#[derive(Debug, Clone)]
pub struct TopologyGene {
    pub innovation_number: String,
    pub pin: NeuronType, //stolen from python-neat for outside connections
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub genetic_weights: Vec<f32>,
    pub genetic_bias: f32,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct NeuronalPropertiesGene {
    pub innovation_number: String,
    pub tau: f32,
    pub homeostatic_force: f32,
    pub tanh_alpha: f32,
}

#[derive(Debug, Clone)]
pub struct MetaLearningGene {
    pub innovation_number: String,
    pub learning_rate: f32,
    pub learning_threshold: f32
}
