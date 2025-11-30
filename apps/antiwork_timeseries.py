from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, from_unixtime, date_format, count
import sys

if len(sys.argv) < 3:
    print("Usage: antiwork_timeseries.py <comments_clean_parquet> <posts_csv>")
    sys.exit(1)

comments_path = sys.argv[1]
posts_path = sys.argv[2]

spark = SparkSession.builder.appName("Antiwork-TimeSeries").getOrCreate()

# =======================
# 1) LOAD COMMENTS (Parquet đã ETL)
# =======================
comments = spark.read.parquet(comments_path)

# comments_clean schema có:
# created_utc_ts, id, subreddit, score, body

comments_month = comments \
    .withColumn("month", date_format(col("created_utc_ts"), "yyyy-MM")) \
    .groupBy("month") \
    .agg(count("*").alias("num_comments"))


# ========================
# 2) LOAD POSTS (CSV dạng raw)
# ========================
posts = spark.read.format("csv") \
    .option("header", "true") \
    .option("multiLine", "true") \
    .option("escape", "\"") \
    .option("quote", "\"") \
    .load(posts_path)

# Needed columns: created_utc, id, title, score
posts = posts.select(
    col("id"),
    col("title"),
    col("score").cast("long"),
    col("created_utc")
)

posts = posts.withColumn(
    "created_ts",
    to_timestamp(from_unixtime(col("created_utc").cast("long")))
)

posts_month = posts \
    .withColumn("month", date_format(col("created_ts"), "yyyy-MM")) \
    .groupBy("month") \
    .agg(count("*").alias("num_posts"))


# =========================
# 3) JOIN COMMENTS + POSTS
# =========================
timeline = comments_month.join(posts_month, on="month", how="outer") \
    .orderBy("month")

print("===== TĂNG TRƯỞNG HOẠT ĐỘNG THEO THÁNG =====")
timeline.show(50, truncate=False)

# =========================
# 4) TÌM THÁNG HOẠT ĐỘNG CAO NHẤT
# =========================
top_month = timeline.orderBy(
    (col("num_comments") + col("num_posts")).desc()
).first()

print("===== THÁNG HOẠT ĐỘNG CAO NHẤT =====")
print(top_month)

spark.stop()
#docker exec -it spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark-apps/antiwork_timeseries.py hdfs://namenode:8020/output/antiwork_cleaned hdfs://namenode:8020/input/posts.csv