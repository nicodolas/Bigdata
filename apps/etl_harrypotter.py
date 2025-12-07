from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.window import Window
import re

spark = SparkSession.builder.appName("ETL-HarryPotter").getOrCreate()
sc = spark.sparkContext

lines = sc.textFile("hdfs://namenode:8020/input/harrypotter.txt")
lines_with_index = lines.zipWithIndex()

def detect_chapter(line):
    return bool(re.match(r"chapter\s+[a-z\-]+", line.lower().strip()))

NUM_MAP = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
    "twenty-three": 23, "twenty-four": 24, "twenty-five": 25,
    "thirty": 30, "thirty-one": 31
}

def extract_chapter_number(line):
    parts = line.lower().strip().split()
    if len(parts) >= 2:
        word = parts[1]
        return NUM_MAP.get(word, None)
    return None

current_chapter = None
def assign_chapter(row):
    global current_chapter
    line, idx = row
    if detect_chapter(line):
        num = extract_chapter_number(line)
        if num:
            current_chapter = num
    return (current_chapter, line)

chapter_rdd = lines_with_index.map(assign_chapter).filter(
    lambda x: x[0] is not None and x[1].strip() != ""
)

chapter_df = spark.createDataFrame(chapter_rdd, ["chapter_number", "raw_text"])

def split_sentences(text):
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

sentences_rdd = chapter_df.rdd.flatMap(
    lambda row: [(row["chapter_number"], s) for s in split_sentences(row["raw_text"])]
)

sentences_df = spark.createDataFrame(sentences_rdd, ["chapter_number", "sentence_text"])

window = Window.partitionBy("chapter_number").orderBy(monotonically_increasing_id())

final_df = sentences_df.withColumn(
    "sentence_number",
    row_number().over(window)
).select("chapter_number", "sentence_number", "sentence_text")

final_df.write.mode("overwrite").parquet(
    "hdfs://namenode:8020/output/harrypotter_structured"
)

spark.stop()
