from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("QuickCheck").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
df = spark.read.parquet("hdfs://namenode:8020/data/processed/cifake_features")
total = df.count()
print(f"TOTAL_FEATURES: {total}")
labels = df.groupBy("label").count().collect()
for r in labels:
    print(f"LABEL_{r['label']}: {r['count']}")
spark.stop()
