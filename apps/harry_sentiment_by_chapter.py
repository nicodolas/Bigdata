from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, udf, avg
from pyspark.sql.types import FloatType

positive = ["good","great","love","happy","kind","safe","brave"]
negative = ["bad","sad","angry","fear","hate","kill","danger"]

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

spark = SparkSession.builder.appName("HarrySentimentChapter").getOrCreate()

df = spark.read.parquet("hdfs://namenode:8020/output/harrypotter_structured")

df = df.withColumn("sentiment", sentiment_udf(col("sentence_text")))

harry = df.filter(lower(col("sentence_text")).like("%harry%"))

chapter_df = (
    harry.groupBy("chapter_number")
         .agg(avg("sentiment").alias("avg_sentiment"))
         .orderBy("chapter_number")
)

chapter_df.write.mode("overwrite").option("header", True).csv("/tmp/harry_chapter_sentiment")

print("DONE")
spark.stop()
