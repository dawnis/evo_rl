pub mod neuron;
pub mod enecode;
pub mod graph;
pub mod doctest;
pub mod population;

use std::collections::HashMap;
use polars::prelude::*;
use std::path::PathBuf;
use std::f32::consts::E;

pub fn setup_logger() {
    pretty_env_logger::try_init().ok();
}

pub fn hash_em(names: Vec<&str>, weights: Vec<f32>) -> HashMap<String, f32> {
    let mut hm: HashMap<String, f32> = HashMap::new();
    for (inn_number, weight) in names.iter().zip(weights.iter()) {
        hm.insert(String::from(*inn_number), *weight);
    }

    hm
}

pub fn ez_input(names: Vec<&str>) -> Vec<String> {
    names.iter().map(|&n| String::from(n)).collect()
}

//Thank you Akshay Ballal for sigmoid and relu
pub fn sigmoid(z: &f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

pub fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {

    let data = CsvReader::from_path(file_path)?.has_header(true).finish()?;

    let training_dataset = data.drop("y")?;
    let training_labels = data.select(["y"])?;

    return Ok((training_dataset, training_labels));

}

//pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
// df.to_ndarray::<Float32Type>().unwrap().reversed_axes()
//}
