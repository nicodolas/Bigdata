from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import re

spark = SparkSession.builder.appName("ETL-HarryPotter").getOrCreate()

# Đọc file text thô
rdd = spark.read.text("hdfs://namenode:8020/input/harrypotter.txt").rdd.map(lambda r: r[0])

# Ghép toàn bộ thành một chuỗi
full = " ".join(rdd.collect())
full = re.sub(r"\s+", " ", full)

# Tách theo CHAPTER
parts = re.split(r"(CHAPTER\s+[A-Za-z0-9]+)", full)

rows = []
chapter_number = 0

for i in range(1, len(parts), 2):
    chapter_number += 1
    chapter_text = parts[i] + " " + parts[i+1]

    # Tách câu đúng bằng regex NLP
    sentences = re.split(r'(?<=[.!?])\s+', chapter_text)

    for idx, s in enumerate(sentences, 1):
        s_clean = s.strip()
        if s_clean:
            rows.append((chapter_number, idx, s_clean))

schema = StructType([
    StructField("chapter_number", IntegerType(), False),
    StructField("sentence_number", IntegerType(), False),
    StructField("sentence_text", StringType(), False),
])

df = spark.createDataFrame(rows, schema)

df.write.mode("overwrite").parquet("hdfs://namenode:8020/output/harrypotter_structured")

print("ETL done.")
spark.stop()
