use crate::neuron::Nn;
use crate::enecode::{EneCode, NeuronalEneCode};
use std::collections::HashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Dfs;

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

    pub fn fwd(self, input: Vec<f32>) -> Vec<f32> {
        // Create a Dfs iterator starting from node `i01`
        let init_node = self.node_identity_map["i01"];
        let mut dfs = Dfs::new(&self.graph, init_node);

        // Iterate over the nodes in depth-first order
        while let Some(nx) = dfs.next(&self.graph) {
            let result = self.graph[nx].propagate();
        }

        result
    }
}
