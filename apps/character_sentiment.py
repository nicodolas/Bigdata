from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, udf, avg
from pyspark.sql.types import FloatType
from pyspark.sql import Row
positive = ["good", "great", "love", "brave", "happy", "kind", "help", "safe", "calm"]
negative = ["bad", "sad", "angry", "fear", "hate", "kill", "danger", "evil"]

def simple_sentiment(text):
    """Tính điểm cảm xúc cơ bản dựa trên từ tích cực / tiêu cực."""
    if not text:
        return 0.0

    text = text.lower()
    score = 0

    for w in positive:
        if w in text:
            score += 1
    for w in negative:
        if w in text:
            score -= 1

    return float(score) / 5.0 

sentiment_udf = udf(simple_sentiment, FloatType())

spark = SparkSession.builder.appName("CharacterSentiment").getOrCreate()

df = spark.read.parquet("hdfs://namenode:8020/output/harrypotter_structured")

df = df.withColumn("sentiment", sentiment_udf(col("sentence_text")))

characters = ["harry", "ron", "hermione", "malfoy", "snape", "dumbledore"]
results = []


for name in characters:
    subset = df.filter(lower(col("sentence_text")).like(f"%{name}%"))

    if subset.count() == 0:
        results.append(Row(character=name, sentiment=None))
    else:
        avg_sent = subset.agg(avg("sentiment")).first()[0]
        results.append(Row(character=name, sentiment=avg_sent))


result_df = spark.createDataFrame(results)

print("\n===== BẢNG SENTIMENT THEO NHÂN VẬT =====\n")
result_df.show(truncate=False)


ranked_df = result_df.orderBy(col("sentiment").desc())

print("\n===== XẾP HẠNG NHÂN VẬT THEO SENTIMENT TRUNG BÌNH =====\n")
ranked_df.show(truncate=False)

spark.stop()
