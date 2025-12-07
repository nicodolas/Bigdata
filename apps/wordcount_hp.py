from pyspark import SparkContext

sc = SparkContext(appName="WordCount-HarryPotter")

text_file = sc.textFile("hdfs://namenode:8020/input/harrypotter.txt")

counts = (text_file
          .flatMap(lambda line: line.split())
          .map(lambda word: (word.lower(), 1))
          .reduceByKey(lambda a, b: a + b))

for word, count in counts.collect():
    print(f"{word}: {count}")

sc.stop()
