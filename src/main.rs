use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use nalgebra::Vector3;

use evo_rl::Nn;

fn main() {

    let mut test_nn = Nn {
        syn: Vector3::new(0.8, 0.8, 0.8),
        ax: 0.,
        tau: 0.2,
        refractory: 0.,
        thresh: 5.,
        connect: String::from("friends")
    };

    let mut rng = rand::thread_rng();
    let input_range = Uniform::new(0., 1.);

    for iter in 0..100 {
        let input_vec: Vec<f32> = (0..3).map( |_| input_range.sample(&mut rng)).collect();
        let input = Vector3::from_vec(input_vec);
        test_nn.fwd_integrate(input, 1.);
        println!("Iteration: {}, Voltage: {}", iter, test_nn.ax);
    }


}
