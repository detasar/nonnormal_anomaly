import pandas as pd
import numpy as np
from nonnormal_anomaly import detect

def test_basic_anomaly_detection():
    # Normal veriler ve birkaç belirgin aykırı değer
    data = np.concatenate([np.random.randn(100) * 2 + 10, np.array([30, 35, -5])])
    df = pd.DataFrame(data, columns=['value'])
    
    result_df = detect(df, column='value', threshold=3.5)
    
    assert 'is_anomaly' in result_df.columns
    assert 'anomaly_score' in result_df.columns
    
    # Aykırı değerlerin tespit edildiğini doğrula
    anomalies = result_df[result_df['is_anomaly']]
    assert len(anomalies) == 3
    assert all(val in anomalies['value'].values for val in [30, 35, -5])
