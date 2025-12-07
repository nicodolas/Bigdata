from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, avg, count, udf
from pyspark.sql.types import FloatType

# ===== SIMPLE SENTIMENT =====
positive = ["good","great","love","happy","kind","safe","brave","support"]
negative = ["bad","sad","angry","fear","hate","kill","danger","terrible"]

def simple_sentiment(t):
    if not t:
        return 0.0
    t = t.lower()
    score = 0
    for w in positive:
        if w in t: score += 1
    for w in negative:
        if w in t: score -= 1
    return float(score)/5.0

sentiment_udf = udf(simple_sentiment, FloatType())

# ===== Spark job =====
spark = SparkSession.builder.appName("Antiwork-Sentiment-Timeline").getOrCreate()

df = spark.read.parquet("hdfs://namenode:8020/output/antiwork_cleaned")

# Tính sentiment
df = df.withColumn("sentiment", sentiment_udf(col("body")))

# Rút tháng từ timestamp
df = df.withColumn("month", date_format(col("created_utc_ts"), "yyyy-MM"))

# Sentiment trung bình theo tháng
sent_month = df.groupBy("month").agg(avg("sentiment").alias("avg_sentiment"))

# Số bình luận theo tháng
count_month = df.groupBy("month").agg(count("*").alias("comment_count"))

# Gộp 2 bảng
final = sent_month.join(count_month, "month").orderBy("month")
final = final.coalesce(1)

# GHI CSV (KHÔNG SHOW)
final.write.mode("overwrite").option("header", True).csv("/tmp/antiwork_sentiment_timeline")

print("DONE!")
spark.stop()
