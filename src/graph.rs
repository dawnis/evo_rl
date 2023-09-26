use crate::neuron::Nn;
use crate::enecode::EneCode;
use std::collections::HashMap;
use petgraph::graph::DiGraph;

struct FeedForwardNeuralNetwork<'a> {
    genome: EneCode<'a>  
}

impl FeedForwardNeuralNetwork<'_> {

    fn get_network(self) -> DiGraph<Nn, f32> {
        let mut graph = DiGraph::new();
        let mut node_identity_map = HashMap::new();

        for gene in self.genome.topology {

            let node = graph.add_node( Nn::from(gene));
            node_identity_map.entry(gene.innovation_number).or_insert(node.index());

            for (i, connection) in gene.outputs.iter().enumerate() {
                graph.add_edge(node, connection, connection.syn[i]);
            }
        }

        graph
    }
}
