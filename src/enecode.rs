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
pub struct EneCode<'a> {
    pub neuron_id: Vec<&'a str>, //equivalent to innovation_number in TopologyGene
    pub topology: Vec<TopologyGene<'a>>,
    pub neuronal_props: NeuronalPropertiesGene<'a>,
    pub meta_learning: MetaLearningGene<'a>
}

impl<'a> EneCode<'a> {
    pub fn topology_gene(&self, neuron_id: &'a str) -> &TopologyGene {
        let gene = self.topology.iter()
            .find(|&g| g.innovation_number == neuron_id)
            .expect("Innovation Number Not Found in Topology Genome!");

        gene
    }
}

//Genetic Makeup of an Individual Neuron
pub struct NeuronalEneCode<'a> {
    pub neuron_id: &'a str,
    pub topology: TopologyGene<'a>,
    pub properties: NeuronalPropertiesGene<'a>,
    pub meta: MetaLearningGene<'a>,
}

impl<'a> NeuronalEneCode<'a> {
    pub fn new_from_enecode(neuron_id: &'a str, genome: &'a EneCode) -> Self {
        let topology = genome.topology_gene(neuron_id).clone();
        NeuronalEneCode {
            neuron_id: neuron_id.clone(),
            topology: topology,
            properties: genome.neuronal_props.clone(),
            meta: genome.meta_learning.clone(), 
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum NeuronType {
    In,
    Out,
    Hidden,
}

#[derive(Debug, Clone)]
pub struct TopologyGene<'a> {
    pub innovation_number: &'a str,
    pub pin: NeuronType, //stolen from python-neat for outside connections
    pub inputs: Vec<&'a str>,
    pub outputs: Vec<&'a str>,
    pub genetic_weights: Vec<f32>,
    pub genetic_bias: f32,
    pub active: bool,
}

#[derive(Debug, Copy, Clone)]
pub struct NeuronalPropertiesGene<'a> {
    pub innovation_number: &'a str,
    pub tau: f32,
    pub homeostatic_force: f32,
    pub tanh_alpha: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct MetaLearningGene<'a> {
    pub innovation_number: &'a str,
    pub learning_rate: f32,
    pub learning_threshold: f32
}
