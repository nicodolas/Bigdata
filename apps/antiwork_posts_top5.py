from pyspark.sql import SparkSession
from pyspark.sql.functions import col, percentile_approx, from_unixtime, to_timestamp

import sys

if len(sys.argv) < 2:
    print("Usage: antiwork_posts_top5.py <posts_csv_path>")
    sys.exit(1)

input_path = sys.argv[1]

spark = SparkSession.builder.appName("Antiwork-Top-Posts-Score").getOrCreate()

# Read CSV
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("multiLine", "true") \
    .option("escape", "\"") \
    .option("quote", "\"") \
    .load(input_path)

# Select + clean columns
df = df.select(
    col("id"),
    col("title"),
    col("selftext"),
    col("score").cast("long"),
    col("created_utc"),
    col("`subreddit.name`").alias("subreddit"),
    col("permalink"),
    col("url")
)

# Convert timestamp
df = df.withColumn(
    "created_utc_ts",
    to_timestamp(from_unixtime(col("created_utc").cast("long")))
)

# Compute 95th percentile
threshold = df.select(percentile_approx("score", 0.95)).first()[0]

top_posts = df.filter(col("score") >= threshold)

print("===== TOP 5% MOST INTERACTIVE POSTS =====")
top_posts.select("id", "score", "title").show(30, truncate=150)

spark.stop()
