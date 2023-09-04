use evo_rl::Nn;

fn main() {

    let test_nn = Nn {
        syn: vec!(1., 2., 3.),
        ax: 0.,
        resist: 10.,
        tau: 5.,
        connect: String::from("friends")
    };

    test_nn.bkwd();
}
