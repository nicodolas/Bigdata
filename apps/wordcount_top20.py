from pyspark import SparkContext
import re

sc = SparkContext(appName="WordCount-Top20-HarryPotter")

text_file = sc.textFile("hdfs://namenode:8020/input/harrypotter.txt")

words = text_file.flatMap(
    lambda line: re.findall(r"[a-zA-Z]+", line.lower())
)

counts = words.map(lambda word: (word, 1)) \
              .reduceByKey(lambda a, b: a + b)

top20 = counts.sortBy(lambda x: x[1], ascending=False).take(20)

for word, count in top20:
    print(f"{word}: {count}")

sc.stop()
