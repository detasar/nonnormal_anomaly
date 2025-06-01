from pyspark.sql import SparkSession
from pyspark.sql.functions import randn
from nonnormal_anomaly.spark import detect_spark

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("NonNormal Anomaly Detection Example") \
        .master("local[4]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    # 1. Örnek Veri Oluşturma (1 milyon normal dağılımlı nokta)
    n_rows = 1_000_000
    df = spark.range(n_rows).withColumn("value", randn(seed=42) * 10 + 50)

    # 2. Aykırı Değerler Ekleme
    outliers = spark.createDataFrame([(1000.0,), (-500.0,), (2500.0,)], ["value"])
    df_with_anomalies = df.union(outliers)

    # 3. Anomali Tespiti Kütüphanesini Çağırma
    print("Spark üzerinde anomali tespiti başlatılıyor...")
    result_df = detect_spark(
        spark_df=df_with_anomalies,
        column="value",
        threshold=3.5,
        approx_relative_error=0.001
    )
    
    print("Anomali tespiti tamamlandı. Sonuçlar gösteriliyor:")
    result_df.show()

    print("Tespit edilen anomaliler:")
    result_df.filter("is_anomaly = true").show()

    spark.stop()
