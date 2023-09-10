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


    for _iter in 0..100 {
        let test_output = test_nn.fwd_integrate(Vector3::new(1., 1., 1.), 1.);
        println!("Our output is {}", test_output);
    }


}
