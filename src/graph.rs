use crate::neuron::Nn;
use crate::enecode::{EneCode, NeuronalEneCode, NeuronType};
use std::collections::HashMap;
use std::sync::Arc;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::stable_graph::EdgeIndex;
use petgraph::visit::Dfs;

use nalgebra as na;
use na::DVector;

/// `FeedForwardNeuralNetwork` is a struct that represents a directed graph
/// based feed-forward neural network, initialized from an `EneCode` genome.
///
/// The struct encapsulates the genome (genetic blueprint), graph-based network,
/// node-identity mapping, and network output.
///
/// # Fields
/// * `genome` - The genetic blueprint (`EneCode`) of the neural network.
/// * `graph` - The directed graph (`DiGraph`) from `petgraph` that represents the network.
/// * `node_identity_map` - A HashMap mapping each neuron ID (`String`) to its index (`NodeIndex`) in the graph.
/// * `network_output` - A vector holding the output values of the output neurons.
///
/// # Example Usage
/// ```rust
/// # use evo_rl::doctest::GENOME_EXAMPLE;
/// use evo_rl::graph::FeedForwardNeuralNetwork;
/// use evo_rl::enecode::EneCode;
///
/// // Assume genome is a properly initialized EneCode
/// # let genome = GENOME_EXAMPLE.clone();
/// let mut network = FeedForwardNeuralNetwork::new(genome);
/// network.initialize();
///
/// // Assume input is a properly initialized Vec<f32>
/// # let input: Vec<f32> = vec![0.];
/// network.fwd(input);
///
/// let output = network.fetch_network_output();
/// ```
#[derive(Debug, Clone)]
pub struct FeedForwardNeuralNetwork {
    genome: EneCode,
    graph: DiGraph<Nn, f32>,
    node_identity_map: HashMap<String, NodeIndex>,
    network_output: Vec<f32>,
}

impl FeedForwardNeuralNetwork {

    /// Create a new `FeedForwardNeuralNetwork` from an `EneCode` genome.
    pub fn new(genome: EneCode) -> Self {
        FeedForwardNeuralNetwork {
            genome: genome.clone(),
            graph: DiGraph::new(),
            node_identity_map: HashMap::new(),
            network_output: Vec::new(),
        }
    }

    /// Initialize the neural network graph from the genome.
    /// Adds neurons as nodes and synaptic connections as edges.
    pub fn initialize(&mut self) {

        //Add all neuron nodes
        for neuron_id in &self.genome.neuron_id[..] {
            let nge = NeuronalEneCode::new_from_enecode(neuron_id.clone(), &self.genome);
            let arc_nge = Arc::new(nge);
            let neuron = Nn::from(arc_nge.clone());
            let node = self.graph.add_node(neuron);
            self.node_identity_map.entry(neuron_id.clone()).or_insert(node);
        }

        //Build Edges
        for daughter in &self.genome.neuron_id[..] {
            let nge = NeuronalEneCode::new_from_enecode(daughter.clone(), &self.genome);
            for parent in nge.topology.inputs.keys() {
                self.graph.add_edge(self.node_identity_map[parent], 
                                    self.node_identity_map[daughter], 
                                    nge.topology.inputs[parent]);
            }
        }

    }

    /// Helper function to identify all input neurons in the network.
    fn fetch_network_input_neurons(&self) -> Vec<NodeIndex> {
        let mut input_ids: Vec<String> = self.genome.neuron_id.iter()
            .filter(|&x| x.starts_with("i"))
            .cloned()
            .collect();

        input_ids.sort();

        input_ids.iter().map(|id| self.node_identity_map[id]).collect()
    }

    /// Forward propagate through the neural network.
    /// This function takes a vector of input values and populates the network output.
    pub fn fwd(&mut self, input: Vec<f32>) {
        // For all input neurons, set values to input
        let input_nodes = self.fetch_network_input_neurons();
        assert_eq!(input.len(), input_nodes.len());

        for (i, &node) in input_nodes.iter().enumerate() {
            self.graph[node].propagate(input[i]);
        }

        // Create a Dfs iterator starting from node `i01`
        let init_node = input_nodes[0]; 
        let mut dfs = Dfs::new(&self.graph, init_node);

        // Create a vector to store the result
        let mut network_output: Vec<f32> = Vec::new();

        // Iterate over the nodes in depth-first order
        while let Some(nx) = dfs.next(&self.graph) {

            let node_parents = self.graph.neighbors_directed(nx, petgraph::Direction::Incoming);

            let mut dot_product: f32 = 0.;

            for pnode in node_parents {


                //grab current synaptic weights
                let edge = self.graph.find_edge(pnode, nx);

                let synaptic_value: f32 = match edge {
                    Some(syn) => *self.graph.edge_weight(syn).expect("Connection was not initialized!"),
                    None => panic!("Improper Edge")
                };

                dot_product = dot_product + synaptic_value * self.graph[pnode].output_value();
            }

            self.graph[nx].propagate(dot_product);

            match self.graph[nx].neuron_type {
                NeuronType::Out => network_output.push( self.graph[nx].output_value() ),
                _ => continue
            }

        }

        self.network_output = network_output;
    }

    /// Fetch the output of the network as a vector of floating-point numbers.
    pub fn fetch_network_output(&self) -> Vec<f32> {
        self.network_output.clone()
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize() {
        // Your test code here
        // Create an EneCode and use it to initialize a FeedForwardNeuralNetwork
        // Validate that the graph is built correctly
    }

    #[test]
    fn test_fwd() {
        // Your test code here
        // Test the forward pass and verify that the network_output is as expected
    }

    #[test]
    fn test_fetch_network_output() {
        // Your test code here
        // Test that the fetch_network_output function returns the expected output
    }
}
