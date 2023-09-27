use crate::neuron::Nn;
use crate::enecode::{EneCode, NeuronalEneCode, NeuronType};
use std::collections::HashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Dfs;

extern crate nalgebra as na;
use na::DVector;

#[derive(Debug, Clone)]
pub struct FeedForwardNeuralNetwork<'a> {
    genome: EneCode<'a>,
    graph: DiGraph<Nn<'a>, ()>,
    node_identity_map: HashMap<&'a str, NodeIndex>,
}

impl<'a> FeedForwardNeuralNetwork<'a> {

    pub fn new<'b: 'a>(genome: &'b EneCode) -> Self {
        FeedForwardNeuralNetwork {
            genome: genome.clone(),
            graph: DiGraph::new(),
            node_identity_map: HashMap::new(),
        }
    }

    pub fn initialize<'b: 'a>(&'b mut self) {

        //Add all neuron nodes
        for neuron_id in &self.genome.neuron_id[..] {
            let nge = NeuronalEneCode::new_from_enecode(neuron_id, &self.genome);
            let neuron = Nn::from(nge);
            let node = self.graph.add_node(neuron);
            self.node_identity_map.entry(neuron_id).or_insert(node);
        }

        //Build Edges
        for neuron_id in &self.genome.neuron_id[..] {
            let nge = NeuronalEneCode::new_from_enecode(neuron_id, &self.genome);
            for daughter in nge.topology.outputs.iter() {
                self.graph.add_edge(self.node_identity_map[neuron_id], self.node_identity_map[daughter], ());
            }
        }

    }

    fn fetch_network_input_neurons(&mut self) -> Vec<NodeIndex> {
        let mut input_ids: Vec<&str> = self.genome.neuron_id.iter()
            .filter(|&&x| x.starts_with("i"))
            .cloned()
            .collect();

        input_ids.sort();

        input_ids.iter().map(|id| self.node_identity_map[id]).collect()
    }

    fn parent_vector(&self, input_list: Vec<&str>) -> DVector<f32> {
        let mut output_vector: Vec<f32> = Vec::new();

        for pid in input_list.iter() {
            let p_node = self.node_identity_map[pid];
            output_vector.push(self.graph[p_node].output_value())
        }

        DVector::from_vec(output_vector)
    }

    pub fn fwd(&mut self, input: Vec<f32>) -> Vec<f32> {
        // For all input neurons, set values to input
        let input_nodes = self.fetch_network_input_neurons();
        assert_eq!(input.len(), input_nodes.len());

        for (i, &node) in input_nodes.iter().enumerate() {
            let input_value_dvec = DVector::from_vec(vec![input[i]]);
            self.graph[node].propagate(input_value_dvec);
        }

        // Create a Dfs iterator starting from node `i01`
        let init_node = input_nodes[0]; 
        let mut dfs = Dfs::new(&self.graph, init_node);

        // Create a vector to store the result
        let mut network_output: Vec<f32> = Vec::new();

        // Iterate over the nodes in depth-first order
        while let Some(nx) = dfs.next(&self.graph) {

            //do nothing for input nodes
            if input_nodes.contains(&nx) {
                continue
            }

            //for all other nodes propagate based on input vector from parent nodes
            let node_input = self.parent_vector(self.graph[nx].inputs.clone());
            self.graph[nx].propagate(node_input);

            match self.graph[nx].neuron_type {
                NeuronType::Out => network_output.push( self.graph[nx].output_value() ),
                _ => continue
            }

        }

        network_output
    }
}
