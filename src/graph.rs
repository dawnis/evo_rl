use crate::neuron::Nn;
use crate::enecode::{EneCode, NeuronalEneCode, NeuronType};
use std::collections::HashMap;
use std::sync::Arc;
use petgraph::graph::{DiGraph, NodeIndex};
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
/// use crate::graph::FeedForwardNeuralNetwork;
/// use crate::enecode::EneCode;
///
/// // Assume genome is a properly initialized EneCode
/// let mut network = FeedForwardNeuralNetwork::new(genome);
/// network.initialize();
///
/// // Assume input is a properly initialized Vec<f32>
/// network.fwd(input);
///
/// let output = network.fetch_network_output();
/// ```
#[derive(Debug, Clone)]
pub struct FeedForwardNeuralNetwork {
    genome: EneCode,
    graph: DiGraph<Nn, ()>,
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
        for neuron_id in &self.genome.neuron_id[..] {
            let nge = NeuronalEneCode::new_from_enecode(neuron_id.clone(), &self.genome);
            for daughter in nge.topology.outputs.iter() {
                self.graph.add_edge(self.node_identity_map[neuron_id], self.node_identity_map[daughter], ());
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

    /// Fetch the output of the network as a vector of floating-point numbers.
    pub fn fetch_network_output(&self) -> Vec<f32> {
        self.network_output.clone()
    }


}
