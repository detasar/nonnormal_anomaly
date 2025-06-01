# Büyük Veri için Yüksek Performanslı, Normal Dağılıma Uymayan Veri Anomali Tespit Kütüphanesi

Bu kütüphane, normal dağılıma uymayan büyük veri kümelerinde anomali tespiti yapmak için Rust dilinde yazılmış yüksek performanslı bir çekirdek ve Python/Spark entegrasyonu sunar. Temel algoritma, aykırı değerlere karşı dayanıklı olan Medyan Mutlak Sapma (MAD) ve Modifiye Edilmiş Z-skoruna dayanmaktadır.

## Kurulum

### Gereksinimler
- Python 3.8+
- Rust programlama dili ve Cargo (https://rustup.rs/ adresinden kurulabilir)

### Kütüphaneyi Kurma

1.  **Depoyu klonlayın:**
    ```bash
    git clone <depo_adresi>
    cd nonnormal_anomaly
    ```

2.  **Sanal bir ortam oluşturun ve etkinleştirin (önerilir):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

3.  **Maturin'i kurun:**
    ```bash
    pip install maturin
    ```

4.  **Kütüphaneyi derleyin ve kurun:**
    Bu komut, Rust kodunu derleyecek ve Python paketini mevcut sanal ortama kuracaktır.
    ```bash
    maturin develop
    ```
    
    Eğer dağıtım için bir wheel dosyası oluşturmak isterseniz:
    ```bash
    maturin build --release
    ```

## Kullanım

### Pandas ile Kullanım

```python
import pandas as pd
from nonnormal_anomaly import detect

# Örnek veri
data = {'sales': [10, 12, 11, 15, 13, 11, 10, 250, 14, 12]}
df = pd.DataFrame(data)

# Anomali tespiti
result = detect(df, column='sales', threshold=3.5)
print(result)
```

### Apache Spark ile Kullanım
Aşağıdaki komutla examples/spark_example.py dosyasını çalıştırabilirsiniz. pyspark'ın kurulu olduğundan emin olun (pip install pyspark).
```bash
spark-submit examples/spark_example.py
```
Bu komut, yerel bir Spark oturumu başlatacak, örnek veri oluşturacak, anomali tespiti yapacak ve sonuçları konsolda gösterecektir.
