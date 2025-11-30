from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, to_timestamp

import sys

if len(sys.argv) < 3:
    print("Usage: etl_antiwork_final.py <input_csv_hdfs_path> <output_parquet_hdfs_path>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

spark = SparkSession.builder.appName("ETL-Antiwork-Final").getOrCreate()

# ==== READ CSV ====
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("multiLine", "true") \
    .option("escape", "\"") \
    .option("quote", "\"") \
    .option("mode", "PERMISSIVE") \
    .load(input_path)

# ==== SELECT & RENAME COLUMNS ====
df = df.select(
    col("id"),
    col("body"),
    col("created_utc"),
    col("score"),
   col("`subreddit.name`").alias("subreddit")
)

# ==== CLEANING ====
# Remove deleted/removed/null
df = df.filter(
    col("body").isNotNull() &
    (~col("body").rlike("(?i)^\\[deleted\\]$")) &
    (~col("body").rlike("(?i)^\\[removed\\]$"))
)

# ==== NORMALIZE TIME ====
df = df.withColumn(
    "created_utc_ts",
    to_timestamp(from_unixtime(col("created_utc").cast("long")))
)

df = df.withColumn("score", col("score").cast("long"))

# ==== FINAL COLUMN ORDER ====
df = df.select(
    "created_utc_ts",
    "id",
    "subreddit",
    "score",
    "body"
)

# ==== WRITE TO HDFS – PARTITIONED BY SUBREDDIT ====
df.write.mode("overwrite").partitionBy("subreddit").parquet(output_path)

print("DONE - Wrote cleaned data to:", output_path)

spark.stop()
