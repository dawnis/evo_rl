use std::collections::HashMap;
use crate::neuron::Nn;

//Evolvable Topological Encoding
// TODO:
// Chromosomal Modules:
//  1. topology and actives/not-active
//  2. integration constants, thresholds, non-linearity parameters
//  3. meta-learning such as synaptic plasticity rules

pub struct NetworkLineage {
    input_dimension: usize,
    architecture: HashMap<String, Nn>,
    division_cycles: usize,
    output_dimension: usize,
    output_fixed: bool,
}
