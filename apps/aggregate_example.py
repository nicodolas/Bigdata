from pyspark.sql import SparkSession

# Tạo SparkSession
spark = SparkSession.builder.appName("AggregateExample").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Tạo dữ liệu
inputRDD = spark.sparkContext.parallelize([("Z", 1), ("A", 20), ("B", 30), ("C", 40), ("B", 30), ("B", 60)])
listRdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5, 3, 2])

# aggregate
param0 = lambda acc, v: acc + v
param1 = lambda acc1, acc2: acc1 + acc2
print("aggregate:", listRdd.aggregate(0, param0, param1))

param3 = lambda acc, v: acc + v[1]
param4 = lambda acc1, acc2: acc1 + acc2
print("aggregate:", inputRDD.aggregate(0, param3, param4))

# treeAggregate
param8 = lambda acc, v: acc + v
param9 = lambda acc1, acc2: acc1 + acc2
print("treeAggregate:", listRdd.treeAggregate(0, param8, param9))

# fold
print("fold:", listRdd.fold(0, lambda acc, v: acc + v))
print("fold:", inputRDD.fold(("Total", 0), lambda acc, v: ("Total", acc[1] + v[1])))

# min, max
print("min:", listRdd.min())
print("min:", inputRDD.min())
print("max:", listRdd.max())
print("max:", inputRDD.max())

# Dừng Spark
spark.stop()
