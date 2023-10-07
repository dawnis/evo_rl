use crate::neuron::Nn;
use crate::enecode::{EneCode, NeuronalEneCode, NeuronType};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::sync::Arc;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Dfs;

/// `NeuralNetwork` is a struct that represents a directed graph
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
/// use evo_rl::graph::NeuralNetwork;
/// use evo_rl::enecode::EneCode;
///
/// // Assume genome is a properly initialized EneCode
/// # let genome = GENOME_EXAMPLE.clone();
/// let mut network = NeuralNetwork::new(genome);
/// network.initialize();
///
/// // Assume input is a properly initialized Vec<f32>
/// # let input: Vec<f32> = vec![0.];
/// network.fwd(input);
///
/// let output = network.fetch_network_output();
/// ```
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    genome: EneCode,
    graph: DiGraph<Nn, f32>,
    node_identity_map: HashMap<String, NodeIndex>,
    network_output: Vec<f32>,
}

impl NeuralNetwork {

    /// Create a new `NeuralNetwork` from an `EneCode` genome.
    pub fn new(genome: EneCode) -> Self {
        NeuralNetwork {
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

    /// Cross over recombination of genetic code
    pub fn recombine_enecode(&self, partner: &NeuralNetwork) -> NeuralNetwork {
        let mut rng = rand::thread_rng();
        let offspring_enecode = self.genome.recombine(&mut rng, &partner.genome);
        match offspring_enecode {
            Ok(enecode) => NeuralNetwork::new(enecode),
            Err(e) => panic!("Recombination error: {:#?}", e)
        }
    }
    
    /// Run mutation for this network
    pub fn mutate(&mut self, mutation_rate: f32) {
        self.mutate_synapses(mutation_rate);
    }

    /// Mutates connections in the network given the current mutation rate
    fn mutate_synapses(&mut self, epsilon: f32) {
        let mut rng = rand::thread_rng();

        //synaptic mutation
        let normal = Normal::new(0., 0.1).unwrap();
        for edge_index in self.graph.edge_indices() {
            if rng.gen::<f32>() < epsilon {
                let new_weight: f32 = self.graph[edge_index] + normal.sample(&mut rng);
                self.graph[edge_index] = if new_weight > 0. {new_weight} else {0.};
            }

        }
    }

    //transfer ownership
    pub fn transfer(self) -> Self {
        self
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

        // Create a Dfs iterator starting from first input node
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
    use crate::doctest::GENOME_EXAMPLE;

    #[test]
    fn test_initialize() {
        let genome = GENOME_EXAMPLE.clone();

        // Create an EneCode and use it to initialize a NeuralNetwork
        let mut network_example = NeuralNetwork::new(genome);
        network_example.initialize();

        // Validate that the graph is built correctly
        let mut dfs = Dfs::new(&network_example.graph, network_example.node_identity_map["input_1"]);

        let mut traversal_order: Vec<String> = Vec::new();

        while let Some(nx) = dfs.next(&network_example.graph) {
            traversal_order.push(network_example.graph[nx].id.clone())
        }

        assert_eq!(vec!["input_1", "N1", "output_1"], traversal_order);
    }

    #[test]
    fn test_fwd_fetch_network_output() {
        let genome = GENOME_EXAMPLE.clone();
        let mut network_example = NeuralNetwork::new(genome);
        network_example.initialize();

        network_example.fwd(vec![0.]);
        // Test the forward pass and verify that the network_output is as expected
        
        let network_out = network_example.fetch_network_output();

        assert_eq!(network_out[0],  0.);
    }

    #[test]
    fn test_mutate_synapses() {
        let genome = GENOME_EXAMPLE.clone();
        let mut network_example = NeuralNetwork::new(genome);
        network_example.initialize();

        let gt = GENOME_EXAMPLE.clone();
        let n1gene = gt.topology_gene(&String::from("N1"));
        let weight_before_mut: f32 = n1gene.inputs["input_1"];

        let epsilon: f32 = 1.;

        network_example.mutate_synapses(epsilon);

        let in1_n1_edge = network_example.graph.find_edge(network_example.node_identity_map["input_1"], network_example.node_identity_map["N1"]);

        let synaptic_value: f32 = match in1_n1_edge {
            Some(syn) => *network_example.graph.edge_weight(syn).expect("Edge not found!!"),
            None => panic!("No weight at edge index")
        };

        assert_ne!(synaptic_value, weight_before_mut);

    }
}
