use numpy::ndarray::{Array3, ArrayView1, ArrayView3};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};


fn uniform(
    voxel_dimensions: ArrayView1<'_, f64>,
    density: ArrayView3<'_, f64>,
    offset: ArrayView1<'_, f64>,
) -> Array3<f64>{
    println!("{:?}", density.len());
    &density + 1.0
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
    ) -> Bound<'py, PyArray3<f64>> {
        uniform(
            voxel_dimensions.as_array(),
            density.as_array(),
            offset.as_array(),
        ).into_pyarray_bound(py)
    }

    Ok(())
}
