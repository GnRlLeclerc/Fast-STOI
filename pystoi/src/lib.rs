use pyo3::prelude::*;

/// Python bindings for stoilib
#[pymodule]
mod stoi {
    use numpy::PyReadonlyArray1;
    use pyo3::{exceptions::PyWarning, prelude::*};

    #[pyfunction]
    fn stoi(
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        fs_sig: u32,
        extended: bool,
    ) -> PyResult<f64> {
        match stoilib::stoi(x.as_array(), y.as_array(), fs_sig, extended) {
            Ok(value) => Ok(value),
            Err(err) => Err(PyWarning::new_err(err.to_string())),
        }
    }
}
