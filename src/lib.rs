use ndarray::prelude::*;
use ndarray_stats::{QuantileExt,interpolate::Linear};
use num_traits::Zero;
use pyo3::prelude::*;
use pyo3_numpy::{PyReadonlyArray1, PyArray1};
use thiserror::Error;

// Özel hata türleri
#[derive(Error, Debug)]
pub enum SpotError {
    #[error("İlk kalibrasyon verisi (n_init) yetersiz: {0} noktadan az olamaz.")]
    InsufficientInitialData(usize),
    #[error("Başlangıç eşiği (level) için %99.9'dan yüksek bir değer seçilemez.")]
    LevelTooHigh,
    #[error("Hesaplama sırasında varyans sıfır çıktı, GPD uydurulamıyor.")]
    ZeroVarianceError,
}

// SPOT algoritmasının durumunu tutan ana yapı (Değişiklik yok)
struct Spot {
    q: f64,
    t: f64,
    num_total: usize,
    num_peaks: usize,
    peaks: Vec<f64>,
    gamma: f64,
    sigma: f64,
}

impl Spot {
    fn new(q: f64, initial_data: ArrayView1<f64>, level: f64) -> Result<Self, SpotError> {
        let n_init = initial_data.len();
        if n_init < 30 { return Err(SpotError::InsufficientInitialData(30)); }
        if level > 0.999 { return Err(SpotError::LevelTooHigh); }

        let mut initial_data_mut = initial_data.to_owned();
        let t = *initial_data_mut.quantile_mut(level, &Linear).unwrap();
        let peaks: Vec<f64> = initial_data.iter().filter(|&&x| x > t).map(|&x| x - t).collect();
        let num_peaks = peaks.len();
        if num_peaks < 2 {
            return Ok(Spot { q, t, num_total: n_init, num_peaks: 0, peaks: vec![], gamma: 0.1, sigma: 1.0 });
        }
        
        let (gamma, sigma) = Self::fit_gpd_mom(&peaks)?;
        Ok(Spot { q, t, num_total: n_init, num_peaks, peaks, gamma, sigma })
    }

    fn fit_gpd_mom(peaks: &[f64]) -> Result<(f64, f64), SpotError> {
        let n = peaks.len() as f64;
        let peak_mean = peaks.iter().sum::<f64>() / n;
        let peak_var = peaks.iter().map(|&p| (p - peak_mean).powi(2)).sum::<f64>() / n;
        if peak_var.abs() < 1e-9 { return Err(SpotError::ZeroVarianceError); }
        let gamma = 0.5 * ((peak_mean.powi(2) / peak_var) - 1.0);
        let sigma = peak_mean * (0.5 + 0.5 * (peak_mean.powi(2) / peak_var));
        Ok((gamma, sigma))
    }

    fn calculate_zq(&self) -> f64 {
        let nt = self.num_peaks as f64;
        if nt == 0.0 { return f64::INFINITY; }
        let n = self.num_total as f64;
        if self.gamma.abs() < 1e-9 { return self.t - self.sigma * (self.q * n / nt).ln(); }
        let term = (self.q * n / nt).powf(-self.gamma);
        self.t + (self.sigma / self.gamma) * (term - 1.0)
    }

    fn process_point(&mut self, x: f64) -> bool {
        self.num_total += 1;
        let zq = self.calculate_zq();
        if x > zq {
            true
        } else if x > self.t {
            self.num_peaks += 1;
            self.peaks.push(x - self.t);
            if let Ok((gamma, sigma)) = Self::fit_gpd_mom(&self.peaks) {
                self.gamma = gamma;
                self.sigma = sigma;
            }
            false
        } else {
            false
        }
    }
}

/// Python'dan çağrılacak ana fonksiyon - ARTIK NUMPY DİZİSİ ALIYOR
#[pyfunction]
fn detect_evt_spot<'py>(
    py: Python<'py>,
    data_array: PyReadonlyArray1<'py, f64>,
    q: f64,
    n_init: usize,
    level: f64,
) -> PyResult<&'py PyArray1<bool>> {
    let data = data_array.as_array(); // Sıfır kopya ile ndarray view'i al

    if data.len() <= n_init {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Veri boyutu, başlangıç boyutu (n_init) kadar veya daha az olamaz."
        ));
    }

    let initial_data = data.slice(s![..n_init]);
    let mut spot_model = Spot::new(q, initial_data, level)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let mut flags: Vec<bool> = vec![false; data.len()];
    for i in n_init..data.len() {
        flags[i] = spot_model.process_point(data[i]);
    }

    Ok(PyArray1::from_vec(py, flags))
}

// Python modülünü oluştur
#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_evt_spot, m)?)?;
    Ok(())
}
