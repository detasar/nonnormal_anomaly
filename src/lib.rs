use arrow_rs::array::{ArrayRef, BooleanArray, Float64Array, PrimitiveArray};
use arrow_rs::datatypes::Float64Type;
use arrow_rs::record_batch::RecordBatch;
use ndarray::prelude::*;
use ndarray_stats::{QuantileExt,interpolate::Linear};
use num_traits::Zero;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_arrow::PyArrowType;
use rayon::prelude::*;

/// Medyan Mutlak Sapma (MAD) ve Modifiye Z-skoru kullanarak anomali tespiti yapar.
///
/// Bu fonksiyon, raporun Bölüm I-B'sinde önerilen dayanıklı istatistiksel yöntemi uygular.
/// Paralel hesaplama için Rayon'dan yararlanır (Bölüm II-B).
///
/// # Arguments
/// * `data` - Girdi verilerini içeren bir ndarray dizisi.
/// * `threshold` - Anomali olarak kabul edilecek Modifiye Z-skoru eşiği (genellikle 3.5).
///
/// # Returns
/// * `(scores, flags)` - Modifiye Z-skorlarını ve anomali bayraklarını içeren bir tuple.
fn detect_anomalies_mad_core(
    data: ArrayView1<f64>,
    threshold: f64,
) -> PyResult<(Vec<f64>, Vec<bool>)> {
    // 1. Verinin medyanını hesapla
    let data_median = data.quantile_mut(0.5, &Linear).unwrap();

    // 2. Medyandan mutlak sapmaları hesapla (paralel olarak)
    let abs_deviations: Vec<f64> = data
        .into_par_iter()
        .map(|x| (x - data_median).abs())
        .collect();
    let mut abs_deviations_arr = Array1::from(abs_deviations);

    // 3. Mutlak sapmaların medyanını (MAD) hesapla
    let mad = abs_deviations_arr.quantile_mut(0.5, &Linear).unwrap();

    // 4. Modifiye Z-skorlarını hesapla
    let scores: Vec<f64> = data
        .into_par_iter()
        .map(|x| {
            // Raporun belirttiği kritik durum: MAD'nin sıfır olması
            if mad.is_zero() {
                if *x == data_median { 0.0 } else { f64::INFINITY }
            } else {
                // Raporun önerdiği ölçekleme faktörü: 0.6745
                0.6745 * (x - data_median) / mad
            }
        })
        .collect();

    // 5. Eşiğe göre anomali bayraklarını belirle
    let flags: Vec<bool> = scores.par_iter().map(|s| s.abs() > threshold).collect();

    Ok((scores, flags))
}

/// Python'dan çağrılacak ana fonksiyon. Apache Arrow verilerini işler.
///
/// Bu fonksiyon, Bölüm III-B'de belirtildiği gibi PyO3 ve pyo3-arrow kullanarak
/// Python ve Rust arasında sıfır kopyalı veri aktarımı sağlar.
#[pyfunction]
fn detect_anomalies_arrow(
    py: Python,
    batch: PyArrowType<RecordBatch>,
    threshold: f64,
) -> PyResult<PyArrowType<RecordBatch>> {
    // 1. Arrow RecordBatch'inden veri sütununu al
    // Sadece ilk sütun üzerinde işlem yapıldığı varsayılıyor.
    let data_col: &PrimitiveArray<Float64Type> = batch
        .as_ref()
        .column(0)
        .as_any()
        .downcast_ref::<PrimitiveArray<Float64Type>>()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected a Float64 column"))?;

    // 2. Arrow verisini ndarray'e dönüştür
    // Raporun IV-B bölümünde belirtildiği gibi, ndarray-stats'in değiştirilebilir
    // dilim gereksinimi nedeniyle bu kopyalama gereklidir.
    let data_view = data_col.values().view();

    // 3. Çekirdek anomali tespit algoritmasını çağır
    let (scores, flags) = detect_anomalies_mad_core(data_view, threshold)?;

    // 4. Sonuçları Arrow dizilerine dönüştür
    let scores_array: ArrayRef = std::sync::Arc::new(Float64Array::from(scores));
    let flags_array: ArrayRef = std::sync::Arc::new(BooleanArray::from(flags));

    // 5. Sonuçları yeni bir RecordBatch olarak Python'a döndür
    let result_batch = RecordBatch::try_from_iter(vec![
        ("scores", scores_array),
        ("is_anomaly", flags_array),
    ])
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyArrowType(result_batch))
}


#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_anomalies_arrow, m)?)?;
    Ok(())
}
