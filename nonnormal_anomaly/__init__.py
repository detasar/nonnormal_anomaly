import pandas as pd
import pyarrow as pa
from typing import Union

# Derlenmiş Rust kütüphanesini içe aktar
from .core import detect_anomalies_arrow

def detect(
    data: Union[pd.DataFrame, pd.Series],
    column: str = None,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Pandas DataFrame veya Series üzerinde anomali tespiti yapar.

    Bu fonksiyon, raporun Bölüm III-A'sında belirtilen kullanıcı dostu
    API tasarımını hedefler.

    Args:
        data: Anomali tespiti yapılacak olan Pandas DataFrame veya Series.
        column: DataFrame kullanılıyorsa işlem yapılacak sütunun adı.
        threshold: Anomali olarak kabul edilecek Modifiye Z-skoru eşiği.

    Returns:
        Orijinal veriye anomali skorlarını ve bayraklarını eklenmiş
        bir Pandas DataFrame.
    """
    if isinstance(data, pd.Series):
        series = data
        df_input = data.to_frame(name=data.name or "value")
    elif isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("DataFrame kullanılıyorsa 'column' parametresi zorunludur.")
        series = data[column]
        df_input = data
    else:
        raise TypeError("Girdi verisi Pandas DataFrame veya Series olmalıdır.")

    # 1. Pandas verisini PyArrow RecordBatch'e dönüştür (verimli)
    record_batch = pa.RecordBatch.from_pandas(series.to_frame(name=series.name))

    # 2. Rust çekirdek fonksiyonunu çağır
    result_batch = detect_anomalies_arrow(record_batch, threshold)

    # 3. Sonucu tekrar Pandas DataFrame'e dönüştür
    result_df = result_batch.to_pandas()

    # 4. Orijinal DataFrame ile sonuçları birleştir
    output_df = df_input.copy()
    output_df['anomaly_score'] = result_df['scores']
    output_df['is_anomaly'] = result_df['is_anomaly']

    return output_df
