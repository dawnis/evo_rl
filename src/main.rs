use nalgebra::Vector3;

use evo_rl::Nn;

fn main() {

    let test_nn = Nn {
        syn: Vector3::new(1., 2., 3.),
        ax: 0.,
        resist: 10.,
        tau: 5.,
        connect: String::from("friends")
    };

    test_nn.bkwd();

    let test_output = test_nn.fwd(Vector3::new(3., 2., 1.));

    println!("Our output is {}", test_output);
}
