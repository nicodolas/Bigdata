from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

positive_words = ["good", "great", "love", "excellent", "amazing", "happy", "support"]
negative_words = ["bad", "terrible", "hate", "awful", "angry", "sad", "against"]

def simple_sentiment(text):
    if text is None:
        return 0
    text = text.lower()
    score = 0
    for w in positive_words:
        if w in text:
            score += 1
    for w in negative_words:
        if w in text:
            score -= 1
    return score

sentiment_udf = F.udf(simple_sentiment, T.IntegerType())

def load_data(spark):
    posts = spark.read.csv(
        "hdfs://namenode:8020/input/posts.csv",
        header=True,
        inferSchema=True
    )
    comments = spark.read.csv(
        "hdfs://namenode:8020/input/comments.csv",
        header=True,
        inferSchema=True
    )
    comments = comments.filter(
        (F.lower(F.col("body")) != "[deleted]") &
        (F.lower(F.col("body")) != "[removed]") &
        (F.col("body").isNotNull())
    )
    posts = posts.withColumn("month", F.date_format(F.from_unixtime("created_utc"), "yyyy-MM"))
    comments = comments.withColumn("month", F.date_format(F.from_unixtime("created_utc"), "yyyy-MM"))
    return posts, comments

def compute_growth(posts, comments):
    posts_by_month = posts.groupBy("month").count().withColumnRenamed("count", "posts")
    comments_by_month = comments.groupBy("month").count().withColumnRenamed("count", "comments")
    monthly = posts_by_month.join(comments_by_month, "month", "outer").fillna(0)
    monthly = monthly.withColumn("total", F.col("posts") + F.col("comments"))
    return monthly.orderBy("month")

def compute_sentiment(comments):
    comments = comments.withColumn("sentiment", sentiment_udf("body"))
    sent_month = comments.groupBy("month").agg(
        F.avg("sentiment").alias("avg_sentiment"),
        F.count("*").alias("count_comments")
    )
    return sent_month.orderBy("month")

def draw_plot(monthly, sent_month):
    pass

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Antiwork-BTVN").getOrCreate()

    posts, comments = load_data(spark)

    monthly = compute_growth(posts, comments)
    monthly.write.mode("overwrite").csv(
        "hdfs://namenode:8020/output/antiwork/monthly_activity",
        header=True
    )

    sent_month = compute_sentiment(comments)
    sent_month.write.mode("overwrite").csv(
        "hdfs://namenode:8020/output/antiwork/monthly_sentiment",
        header=True
    )

    spark.stop()
