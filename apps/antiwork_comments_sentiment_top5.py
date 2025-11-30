from pyspark.sql import SparkSession
from pyspark.sql.functions import col, percentile_approx
import sys

if len(sys.argv) < 2:
    print("Usage: antiwork_comments_sentiment_top5.py <comments_csv_path>")
    sys.exit(1)

input_path = sys.argv[1]

spark = SparkSession.builder.appName("Antiwork-Comments-Sentiment-Top5").getOrCreate()

# Đọc file CSV comments
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("multiLine", "true") \
    .option("escape", "\"") \
    .option("quote", "\"") \
    .load(input_path)

# Lấy đúng các trường cần thiết
df = df.select(
    col("id"),
    col("body"),
    col("sentiment").cast("float")
)

# Bỏ bỏ những dòng không có sentiment
df = df.filter(col("sentiment").isNotNull())

# Tính phân vị
upper = df.select(percentile_approx("sentiment", 0.95)).first()[0]
lower = df.select(percentile_approx("sentiment", 0.05)).first()[0]

# Lọc top 5% sentiment cao nhất & thấp nhất
top_pos = df.filter(col("sentiment") >= upper)
top_neg = df.filter(col("sentiment") <= lower)

print("===== TOP 5% SENTIMENT CAO NHẤT =====")
top_pos.select("id", "sentiment", "body").show(30, truncate=150)

print("===== TOP 5% SENTIMENT THẤP NHẤT =====")
top_neg.select("id", "sentiment", "body").show(30, truncate=150)

spark.stop()
#docker exec -it spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark-apps/antiwork_comments_sentiment_top5.py hdfs://namenode:8020/input/comments.csv
