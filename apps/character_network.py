from pyspark import SparkContext
import re
from itertools import combinations

sc = SparkContext(appName="Character-Network")

# Danh sách nhân vật
characters = ["harry", "ron", "hermione", "malfoy", "snape", "dumbledore"]
char_set = set(characters)

# Đọc file đã ETL hoặc đọc trực tiếp từ HDFS
# Ở đây ta đọc lại file thô và xử lý theo câu:
text = sc.textFile("hdfs://namenode:8020/input/harrypotter.txt")

# Hàm tách câu
def split_sentences(line):
    parts = re.split(r"[.!?]+", line)
    return [p.strip() for p in parts if len(p.strip()) > 0]

sentences = text.flatMap(split_sentences)

# Chuẩn hóa câu về lowercase và tách từ
def extract_characters(sentence):
    words = re.findall(r"[a-zA-Z]+", sentence.lower())
    present = [w for w in words if w in char_set]
    return list(set(present))  # tránh lặp từ trong cùng 1 câu

# Lấy danh sách nhân vật xuất hiện trong mỗi câu
chars_per_sentence = sentences.map(extract_characters)

# Tạo cặp (pair) trong câu
def make_pairs(char_list):
    if len(char_list) < 2:
        return []
    # combinations tạo các cặp không trùng lặp
    return [tuple(sorted(pair)) for pair in combinations(char_list, 2)]

pairs = chars_per_sentence.flatMap(make_pairs)

# Đếm tần suất
pair_counts = pairs.map(lambda p: (p, 1)) \
                   .reduceByKey(lambda a, b: a + b)

# Sắp xếp giảm dần
sorted_pairs = pair_counts.sortBy(lambda x: -x[1])

# Lấy toàn bộ kết quả
results = sorted_pairs.collect()

print("===== MẠNG LƯỚI TƯƠNG TÁC NHÂN VẬT =====")
for pair, count in results:
    print(f"{pair}: {count}")

sc.stop()
