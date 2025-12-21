# Hướng dẫn chạy Step 4: Inference Demo

## Mục đích
Chứng minh pipeline có thể xử lý ảnh mới chưa từng thấy, trích xuất đặc trưng, lưu HDFS và dự đoán.

## Chuẩn bị

### 1. Tải 1 ảnh test về máy
Bạn có thể:
- Tải 1 ảnh bất kỳ từ Internet
- Hoặc dùng ảnh có sẵn trong dataset test

### 2. Copy ảnh vào Docker container
```bash
# Copy ảnh từ máy local vào spark-master container
docker cp /path/to/your/test_image.jpg spark-master:/tmp/test_image.jpg
```

### 3. Upload ảnh lên HDFS (nếu muốn)
```bash
# Vào container
docker exec -it spark-master bash

# Upload lên HDFS
hdfs dfs -put /tmp/test_image.jpg /data/demo/

# Hoặc giữ ở /tmp/ trong container cũng được
```

## Cách chạy

### Bước 1: Sửa đường dẫn trong script
Mở file `step4_inference_demo.py`, sửa dòng:
```python
TEST_IMAGE_PATH = "/tmp/test_image.jpg"  # Đường dẫn trong container
```

### Bước 2: Submit job
```bash
docker exec -it spark-master bash

spark-submit \
  --master spark://spark-master:7077 \
  --executor-memory 3g \
  --driver-memory 4g \
  /apps/step4_inference_demo.py
```

## Kết quả mong đợi

Script sẽ:
1. ✅ Nạp MobileNetV2
2. ✅ Trích xuất vector 1280 chiều
3. ✅ Lưu vào HDFS: `/data/demo/single_image_features` (định dạng Parquet)
4. ✅ Load model đã train từ Step 3
5. ✅ Dự đoán nhãn: REAL hoặc FAKE
6. ✅ Hiển thị confidence score

## Output mẫu
```
============================================================
PREDICTION RESULT
============================================================
Image: test_image.jpg
Prediction: FAKE (Ảnh giả - AI Generated)
Confidence: 94.23%
Probability [FAKE, REAL]: [0.9423, 0.0577]
============================================================

BUSINESS INSIGHT
============================================================
Câu hỏi: Liệu model có trích xuất đủ thông tin để phát hiện Deepfake?

Trả lời: CÓ - Dựa trên demo này:
1. Vector 1280 chiều đã được trích xuất thành công từ ảnh mới
2. Dữ liệu đã được lưu vào HDFS dưới định dạng Parquet
3. Model có thể load và dự đoán chính xác
4. Độ tin cậy cao chứng minh thông tin đặc trưng đủ phân biệt
============================================================
```

## Kiểm tra kết quả trên HDFS
```bash
# Xem cấu trúc file Parquet
hdfs dfs -ls /data/demo/single_image_features/

# Đọc nội dung (nếu muốn)
spark-shell --master spark://spark-master:7077

scala> val df = spark.read.parquet("hdfs://namenode:8020/data/demo/single_image_features")
scala> df.show(false)
```

## Lưu ý quan trọng
- Script này chạy **NGOÀI pipeline chính** (Step 1-2-3)
- Mục đích: Demo khả năng inference real-time
- Kết quả được lưu vào thư mục riêng `/data/demo/` để không ảnh hưởng dữ liệu train
