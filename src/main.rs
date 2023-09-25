use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use nalgebra::Vector3;

use evo_rl::neuron::Nn;

fn main() {
    //Immediate Goal: Evolve a neural network to solve XOR? 

    let mut test_nn = Nn {
        syn: Vector3::new(0.1, 0.2, 0.4),
        ax: 0.,
        tau: 0.2,
        learning_threshold: 0.5,
        learning_rate: 0.001,
        alpha: 1.0,
    };

    let mut rng = rand::thread_rng();
    let input_range = Uniform::new(0., 1.);

    for iter in 0..100 {
        let input_vec: Vec<f32> = (0..3).map( |_| input_range.sample(&mut rng)).collect();
        let input = Vector3::from_vec(input_vec);
        test_nn.fwd(input);
        println!("Iteration: {}, Voltage: {}, Weights: {}", iter, test_nn.ax, test_nn.syn);
    }


}
