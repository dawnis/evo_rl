pub mod neuron;
pub mod enecode;
pub mod graph;
pub mod doctest;
pub mod population;

use log::*;
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

/// Increments the ID of a neuron when creating a daughter
pub fn increment_innovation_number(neuron_id: &String, daughter_ids: Vec<&String>) -> String {
    //innovation numbers will be of the form alphanumeric string (progenitor code) followed by
    //numeric (lineage code)
    //First, identify the progenitor code
    
    let progenitor_code: &str = match neuron_id.find("-") {
        Some(idx) => {
        let (pc, _tail) = neuron_id.split_at(idx);
        pc
        },
        None => neuron_id
    };

    let daughter_ids_progenitor: Vec<&&String> = daughter_ids.iter().filter(|&id| id.starts_with(progenitor_code)).collect();

    //If it is the first daughter, add -1 to the end of the string
    if daughter_ids_progenitor.len() == 0 {
        let mut daughter_id = neuron_id.clone();
        daughter_id.push_str("-1");
        daughter_id
    } else {
        //else increment the largest daughter
        let largest_daughter_id = daughter_ids_progenitor.iter().max().unwrap();

        if let Some(idx) = largest_daughter_id.rfind("-") {
            let (previous_lineage, largest_daughter_number) = largest_daughter_id.split_at(idx);

            let ldn: i32 = match largest_daughter_number[1..].parse() {
                Ok(n) => n,
                Err(e) => panic!("Failed to parse string daughter number"),
            };

            let mut daughter_id = String::from(previous_lineage);
            daughter_id.push('-');
            daughter_id.push_str(&(ldn + 1).to_string());

            daughter_id.to_string()

        } else {
            debug!("Problem with parsing string largest_daughter_id {} while duplicating {}", largest_daughter_id, neuron_id);
            panic!("Attempted to parse daughter innovation number but found invalid code");
        }
    }
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
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increment_innovation_number() {
        let innovation_number = String::from("a0");
        let daughters = Vec::new();

        let d1 = increment_innovation_number(&innovation_number, daughters);
        assert_eq!(d1, String::from("a0-1"));


        let a01 = String::from("a0-1");
        let a02 = String::from("a0-2");
        let daughters2 = vec![&a01, &a02];
        let d2 = increment_innovation_number(&innovation_number, daughters2);
        assert_eq!(d2, String::from("a0-3"));

        let innovation_number2 = String::from("a0-2-2");
        let a03 = String::from("a0-2-2-1");
        let a04 = String::from("a0-2-2-20");
        let b01 = String::from("B0-10000");

        let daughters3 = vec![&a03, &a04, &b01];
        let d3 = increment_innovation_number(&innovation_number2, daughters3);

        assert_eq!(d3, String::from("a0-2-2-21"));
    }
}
