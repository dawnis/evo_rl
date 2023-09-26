use crate::neuron::Nn;
use crate::enecode::{EneCode, NeuronalEneCode};
use std::collections::HashMap;
use petgraph::graph::{DiGraph, NodeIndex};

#[derive(Debug, Clone)]
struct FeedForwardNeuralNetwork<'a> {
    genome: EneCode<'a>,
    graph: DiGraph<Nn<'a>, ()>,
    node_identity_map: HashMap<&'a str, NodeIndex>,
}

impl<'a> FeedForwardNeuralNetwork<'a> {

    fn new<'b: 'a>(genome: &'b EneCode) -> Self {
        FeedForwardNeuralNetwork {
            genome: genome.clone(),
            graph: DiGraph::new(),
            node_identity_map: HashMap::new(),
        }
    }

    fn initialize<'b: 'a>(&'b mut self) {

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
}
