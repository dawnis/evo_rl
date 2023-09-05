use nalgebra::Vector3;

pub fn relu(z: &f32) -> f32 {
    match *z > 0. {
        true => *z,
        false => 0.
    }
}

pub struct Nn {
    pub syn: Vector3<f32>,
    pub ax: f32,
    pub tau: f32,
    pub refractory: f32,
    pub thresh: f32,
    pub connect: String
}

impl Nn {

    fn fwd(&mut self, input: Vector3<f32>) {
        let impulse: f32 = input.dot(&self.syn);
        self.ax = self.ax * (-self.tau).exp() + impulse;
    }

    fn learn(&self) {
        println!("I'm learning mom!")
    }

    fn reset(&mut self) {
        self.refractory = 2.
    }

    pub fn fire(&mut self, input: Vector3<f32>, time_delta: f32) -> f32 {
        self.fwd(input);
        self.refractory = self.refractory - time_delta;
        if self.ax > self.thresh & self.refractory <= 0 {
            self.learn();
            relu(&self.ax)
        } else { 0. }

    }

}
