use numpy::{PyReadonlyArray1}
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn uniform(

)

/// A Python module implemented in Rust.
#[pymodule]
fn _algorithms(m: &Bound<'_, PyModule>) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name="uniform")]
    fn uniform_py(
        PyReadonlyArray1<f64> voxel_dimensions,

    )

    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
