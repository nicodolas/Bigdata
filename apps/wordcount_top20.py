from pyspark import SparkContext
import re

sc = SparkContext(appName="WordCount-Top20-HarryPotter")

# Đọc dữ liệu từ HDFS
text_file = sc.textFile("hdfs://namenode:8020/input/harrypotter.txt")

# Tách từ, làm sạch ký tự, chuyển về chữ thường
words = text_file.flatMap(
    lambda line: re.findall(r"[a-zA-Z]+", line.lower())
)

# Đếm số lần xuất hiện
counts = words.map(lambda word: (word, 1)) \
              .reduceByKey(lambda a, b: a + b)

# Sắp xếp giảm dần theo số lần xuất hiện và lấy Top 20
top20 = counts.sortBy(lambda x: x[1], ascending=False).take(20)

# In kết quả
print("===== TOP 20 TỪ XUẤT HIỆN NHIỀU NHẤT =====")
for word, count in top20:
    print(f"{word}: {count}")

sc.stop()
