use crate::neuron::Nn;
use crate::enecode::{EneCode, NeuronalEneCode, NeuronType};
use std::collections::HashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Dfs;

extern crate nalgebra as na;
use na::DVector;

#[derive(Debug, Clone)]
pub struct FeedForwardNeuralNetwork {
    genome: EneCode,
    graph: DiGraph<Nn, ()>,
    node_identity_map: HashMap<String, NodeIndex>,
    network_output: Vec<f32>,
}

impl FeedForwardNeuralNetwork {

    pub fn new(genome: EneCode) -> Self {
        FeedForwardNeuralNetwork {
            genome: genome.clone(),
            graph: DiGraph::new(),
            node_identity_map: HashMap::new(),
            network_output: Vec::new(),
        }
    }

    pub fn initialize(&mut self) {

        //Add all neuron nodes
        for neuron_id in &self.genome.neuron_id[..] {
            let nge = NeuronalEneCode::new_from_enecode(neuron_id.clone(), &self.genome);
            let neuron = Nn::from(nge);
            let node = self.graph.add_node(neuron);
            self.node_identity_map.entry(neuron_id.clone()).or_insert(node);
        }

        //Build Edges
        for neuron_id in &self.genome.neuron_id[..] {
            let nge = NeuronalEneCode::new_from_enecode(neuron_id.clone(), &self.genome);
            for daughter in nge.topology.outputs.iter() {
                self.graph.add_edge(self.node_identity_map[neuron_id], self.node_identity_map[daughter], ());
            }
        }

    }

    fn fetch_network_input_neurons(&self) -> Vec<NodeIndex> {
        let mut input_ids: Vec<String> = self.genome.neuron_id.iter()
            .filter(|&x| x.starts_with("i"))
            .cloned()
            .collect();

        input_ids.sort();

        input_ids.iter().map(|id| self.node_identity_map[id]).collect()
    }

    pub fn fwd(&mut self, input: Vec<f32>) {
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

            let mut ip: Vec<f32> = Vec::new();

            for pid in self.graph[nx].inputs.iter() {
                let p_node = self.node_identity_map[pid];
                ip.push(self.graph[p_node].output_value())
            }

            self.graph[nx].propagate(DVector::from_vec(ip));

            match self.graph[nx].neuron_type {
                NeuronType::Out => network_output.push( self.graph[nx].output_value() ),
                _ => continue
            }

        }

        self.network_output = network_output;
    }

    pub fn fetch_network_output(&self) -> Vec<f32> {
        self.network_output.clone()
    }


}
