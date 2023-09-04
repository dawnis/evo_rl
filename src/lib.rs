pub struct Nn {
    pub syn: Vec<f32>,
    pub ax: f32,
    pub tau: f32,
    pub resist: f32,
    pub connect: String
}

impl Nn {

    pub fn fwd(&self) -> f32 {
        1.
    }

    pub fn bkwd(&self) {
        println!("I'm going backwards mom!")
    }
}
