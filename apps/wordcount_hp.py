from pyspark import SparkContext

sc = SparkContext(appName="WordCount-HarryPotter")

# Đọc file từ HDFS
text_file = sc.textFile("hdfs://namenode:8020/input/harrypotter.txt")

# Word Count
counts = (text_file
          .flatMap(lambda line: line.split())
          .map(lambda word: (word.lower(), 1))
          .reduceByKey(lambda a, b: a + b))

# Lấy kết quả về và in ra
output = counts.collect()

for word, count in output:
    print(f"{word}: {count}")

sc.stop()
