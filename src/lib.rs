use nalgebra::Vector3;

pub struct Nn {
    pub syn: Vector3<f32>,
    pub ax: f32,
    pub tau: f32,
    pub resist: f32,
    pub connect: String
}

impl Nn {

    pub fn fwd(&self, input: Vector3<f32>) -> f32 {
        input.dot(&self.syn)
    }

    pub fn bkwd(&self) {
        println!("I'm going backwards mom!")
    }
}
