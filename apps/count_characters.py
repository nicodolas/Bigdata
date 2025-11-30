from pyspark import SparkContext
import re

sc = SparkContext(appName="Count-Harry-Characters")

# Danh sách các từ cần đếm
target_words = ["harry", "ron", "hermione", "malfoy", "snape", "dumbledore"]

# Broadcast để tối ưu khi chạy phân tán
broadcast_targets = sc.broadcast(set(target_words))

# Đọc dữ liệu từ HDFS
text_file = sc.textFile("hdfs://namenode:8020/input/harrypotter.txt")

# Tách từ, chuẩn hóa
words = text_file.flatMap(
    lambda line: re.findall(r"[a-zA-Z]+", line.lower())
)

# Chỉ giữ lại các từ cần đếm
filtered_words = words.filter(lambda w: w in broadcast_targets.value)

# Đếm số lần xuất hiện
counts = filtered_words.map(lambda w: (w, 1)) \
                        .reduceByKey(lambda a, b: a + b)

# Gom kết quả về driver
result = dict(counts.collect())

# In đầy đủ, kể cả từ có thể = 0
print("===== SỐ LẦN XUẤT HIỆN CÁC NHÂN VẬT =====")
for name in target_words:
    print(f"{name}: {result.get(name, 0)}")

sc.stop()
