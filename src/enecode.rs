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
    pub topology: Vec<ToplogyGene<'a>>,
    pub neuronal_props: NeuronalPropertiesGene<'a>,
    pub meta_learning: MetaLearningGene<'a>
}

impl<'a> EneCode<'a> {
    pub fn topology_gene(&self, neuron_id: &'a str) -> &ToplogyGene {
        let gene = self.topology.iter()
            .find(|&g| g.innovation_number == neuron_id)
            .expect("Innovation Number Not Found in Topology Genome!");

        gene
    }
}

//Genetic Makeup of an Individual Neuron
pub struct NeuronalEneCode<'a> {
    pub neuron_id: &'a str,
    pub topology: &'a ToplogyGene<'a>,
    pub properties: &'a NeuronalPropertiesGene<'a>,
    pub meta: &'a MetaLearningGene<'a>,
}

impl<'a> NeuronalEneCode<'a> {
    pub fn new_from_enecode(neuron_id: &'a str, genome: &'a EneCode) -> Self {
        NeuronalEneCode {
            neuron_id,
            topology: genome.topology_gene(neuron_id),
            properties: &genome.neuronal_props,
            meta: &genome.meta_learning, 
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToplogyGene<'a> {
    pub innovation_number: &'a str,
    pub pin: usize, //stolen from python-neat for outside connections
    pub inputs: Vec<&'a str>,
    pub outputs: Vec<&'a str>,
    pub genetic_weights: Vec<f32>,
    pub active: bool,
}

#[derive(Debug, Copy, Clone)]
pub struct NeuronalPropertiesGene<'a> {
    pub innovation_number: &'a str,
    pub tau: f32,
    pub homeostatic_force: f32,
    pub alpha: f32
}

#[derive(Debug, Copy, Clone)]
pub struct MetaLearningGene<'a> {
    pub innovation_number: &'a str,
    pub learning_rate: f32,
    pub learning_threshold: f32
}
