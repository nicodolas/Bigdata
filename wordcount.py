from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("WordCount").setMaster("spark://spark-master:7077")
sc = SparkContext(conf=conf)

data = ["hello spark", "hello hadoop", "spark is great"]
rdd = sc.parallelize(data)
counts = (rdd.flatMap(lambda line: line.split(" "))
              .map(lambda word: (word, 1))
              .reduceByKey(lambda a, b: a + b))

for word, count in counts.collect():
    print(f"{word}: {count}")

sc.stop()
