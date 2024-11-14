use numpy::ndarray::{ArrayD, ArrayViewD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};


fn uniform(
    voxel_dimensions: ArrayViewD<'_, f64>,
    density: ArrayViewD<'_, f64>,
    offset: ArrayViewD<'_, f64>,
) -> ArrayD<f64>{
    &density + 1.0
}


/// A Python module implemented in Rust.
#[pymodule]
fn _algorithms_impl<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "uniform")]
    fn uniform_py<'py>(
        py: Python<'py>,
        voxel_dimensions: PyReadonlyArrayDyn<'py, f64>,
        density: PyReadonlyArrayDyn<'py, f64>,
        offset: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        uniform(
            voxel_dimensions.as_array(),
            density.as_array(),
            offset.as_array(),
        ).into_pyarray_bound(py)
    }

    Ok(())
}
