use evo_rl::enecode::{EneCode, TopologyGene, NeuronalPropertiesGene, MetaLearningGene, NeuronType};
use evo_rl::graph::FeedForwardNeuralNetwork;

extern crate rand;
use rand::distributions::{Distribution, Bernoulli};

struct Foo<'a> {
    value: Vec<&'a str>,
}

impl<'a> Foo<'a> {
    // Initialization function that needs a mutable reference
    fn initialize(&mut self) {
        self.value.push("hello");
    }

    // Update function that also needs a mutable reference
    fn update(&mut self) {
        self.value.push("world");
    }

    // Function to read the value
    fn read(&self) {
        println!("Value is: {:?}", self.value);
    }
}

fn main() {
    //Immediate Goal: Evolve a neural network to solve XOR? 
    //
    //1. Define genetic structur for XOR minimal network, including inputs and outputs
    //2. Test random and init over 1000 turns using backprop
    //3. Move this to a unit Test

    let mut foo = Foo { value: Vec::new() };

    // First mutable borrow for initialization
    foo.initialize();

    // Second mutable borrow for updating, after the first has been released
    foo.update();

    // Now we can borrow it immutably to read the value
    foo.read(); // Should output: Value is: ["hello", "world"]

    let xor_network_genome = EneCode {
        neuron_id: vec!["i01", "i02", "D", "E"],
        topology: vec![
            TopologyGene {
                innovation_number: "i01",
                pin: NeuronType::In,
                inputs: vec![],
                outputs: vec!["D", "E"],
                genetic_weights: vec![],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "i02",
                pin: NeuronType::In,
                inputs: vec![],
                outputs: vec!["D", "E"],
                genetic_weights: vec![],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "D",
                pin: NeuronType::Hidden,
                inputs: vec!["b", "i01", "i02"],
                outputs: vec!["E"],
                genetic_weights: vec![1., 0.2, 0.3],
                genetic_bias: 0.,
                active: true,
            },

            TopologyGene {
                innovation_number: "E",
                pin: NeuronType::Out,
                inputs: vec!["b", "i01", "i02", "D"],
                outputs: vec![],
                genetic_weights: vec![1., -0.1, 0.3],
                genetic_bias: 0.,
                active: true,
            },
        ],
        neuronal_props: NeuronalPropertiesGene {
            innovation_number: "p01",
            tau: 0.,
            homeostatic_force: 0.,
            tanh_alpha: 1.,
        },
        meta_learning: MetaLearningGene {
            innovation_number: "m01",
            learning_rate: 0.001,
            learning_threshold: 0.5,
        }

    };


    let mut fnn = FeedForwardNeuralNetwork::new(&xor_network_genome);


    let bernoulli = Bernoulli::new(0.5).unwrap(); // Create a Bernoulli distribution with p=0.5
    let sample: Vec<bool> = (0..2).into_iter().map(|_| bernoulli.sample(&mut rand::thread_rng())).collect();

    fnn.initialize(); 

    fnn.fwd(vec![0., 1.]);

    let xor_actual = false ^ true;

    let network_evaluation = fnn.fetch_network_output();

    println!("Network evaluation was {} and XOR result was {}!", network_evaluation[0], xor_actual);

}
