use rand::{Rng};

use numpy::ndarray::{Array2, Array3, ArrayView1, ArrayView3};
use numpy::{IntoPyArray, PyArray1, PyArray2,  PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};


fn choose_voxel_index(
    uniform_value: f64,
    cumulative_probabilities: &Vec<f64>,
    voxel_indices: &Vec<usize>
) -> usize {
    cumulative_probabilities
        .binary_search_by(
            |p| p.partial_cmp(&uniform_value).unwrap_or(std::cmp::Ordering::Greater)
        )
        .unwrap_or_else(|_| 0)
}


fn uniform(
    voxel_dimensions: ArrayView1<'_, f64>,
    density: ArrayView3<'_, f64>,
    offset: ArrayView1<'_, f64>,
) -> Array2<f64>{
    println!("{:?}", density.len());

    let factor = 1e-9 * voxel_dimensions.iter().copied().product::<f64>();

    let (nonzero_voxel_indices, nonzero_voxel_counts): (Vec<_>, Vec<_>)= density
        .iter()
        .enumerate()
        .filter_map(|(i, &value)|
            if value > 0.0 {
                Some((i, value * factor))
            } else {
                None
            }
        )
        .unzip();

    let total_count: f64 = nonzero_voxel_counts.iter().sum();

    let cumulative_probabilities = nonzero_voxel_counts
        .iter()
        .scan(0.0, |sum, &count| {
                *sum += count / total_count;
                Some(*sum)
            }
        )
        .collect();

    let mut rng = rand::thread_rng();

    let n_positions = total_count.round() as usize;

    let nx = density.shape()[0];
    let ny = density.shape()[1];
    let nz = density.shape()[2];

    let mut positions = Array2::<f64>::zeros((n_positions, 3));

    for index in 0..n_positions {

        let uniform_sample = rng.gen::<f64>();

        let voxel_index = choose_voxel_index(
            uniform_sample,
            &cumulative_probabilities,
            &nonzero_voxel_indices,
        );

        let i: usize = voxel_index % nx;
        let j: usize = ((voxel_index - i) / nx ) % ny;
        let k: usize = ((voxel_index - i) / nx - j) / &ny; 
      
        positions.row_mut(index)[0] = offset[0] + i as f64 + rng.gen::<f64>();
        positions.row_mut(index)[1] = offset[1] + j as f64 + rng.gen::<f64>();
        positions.row_mut(index)[2] = offset[2] + k as f64 + rng.gen::<f64>();

    }

    positions

}


/// A Python module implemented in Rust.
#[pymodule]
fn _algorithms_impl<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "uniform")]
    fn uniform_py<'py>(
        py: Python<'py>,
        voxel_dimensions: PyReadonlyArray1<'py, f64>,
        density: PyReadonlyArray3<'py, f64>,
        offset: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        uniform(
            voxel_dimensions.as_array(),
            density.as_array(),
            offset.as_array(),
        ).into_pyarray_bound(py)
    }

    Ok(())
}
