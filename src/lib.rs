pub mod neuron;
pub mod enecode;
pub mod graph;
pub mod doctest;
pub mod population;

use polars::prelude::*;
use std::path::PathBuf;

pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {

    let data = CsvReader::from_path(file_path)?.has_header(true).finish()?;

    let training_dataset = data.drop("y")?;
    let training_labels = data.select(["y"])?;

    return Ok((training_dataset, training_labels));

}

//pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
// df.to_ndarray::<Float32Type>().unwrap().reversed_axes()
//}
