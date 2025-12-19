from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
import sys
import time

# =====================================================
# Hàm khởi tạo Spark Session
# =====================================================
def create_spark_session():
    spark = SparkSession.builder \
        .appName("CIFAKE-Step-2-Optimized") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


# =====================================================
# Optimized Distributed Feature Extraction (Batch + Quantized)
# =====================================================
def extract_features_optimized(iterator):
    """
    Xử lý theo Partition, dùng Batch Inference và Quantized Model
    """
    import torch
    from torchvision import models, transforms
    from PIL import Image
    import io
    import numpy as np
    
    # -----------------------------------------------
    # 1. Load MobileNetV2 (Float32 - Ổn định trên mọi CPU)
    # -----------------------------------------------
    print("[INFO] Loading MobileNetV2 (Float32)...", file=sys.stderr)
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Identity() # Bỏ lớp cuối
    model.eval()
    print("[OK] Model ready! (1280-dim features)", file=sys.stderr)
    
    # Preprocessing chuẩn
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 64
    batch_images = []
    batch_rows = []
    
    count = 0
    
    with torch.inference_mode(): # Tối ưu hơn no_grad
        for row in iterator:
            try:
                # Load ảnh
                image = Image.open(io.BytesIO(row.content)).convert("RGB")
                input_tensor = preprocess(image)
                
                batch_images.append(input_tensor)
                batch_rows.append(row)
                
                # Nếu đủ batch -> Predict 1 lần
                if len(batch_images) >= batch_size:
                    # Stack thành (64, 3, 224, 224)
                    input_batch = torch.stack(batch_images)
                    
                    # Inference
                    features_batch = model(input_batch)
                    features_list = features_batch.numpy().tolist()
                    
                    # Yield kết quả
                    for i, features in enumerate(features_list):
                        yield (features, int(batch_rows[i].label))
                    
                    # Reset batch
                    batch_images = []
                    batch_rows = []
                    
                    count += batch_size
                    if count % 1000 == 0:
                        print(f"[PROGRESS] Processed {count} images...", file=sys.stderr)
                        
            except Exception as e:
                print(f"[ERROR] {e}", file=sys.stderr)
                continue
        
        # Xử lý batch cuối cùng (nếu còn lẻ)
        if len(batch_images) > 0:
            input_batch = torch.stack(batch_images)
            features_batch = model(input_batch)
            features_list = features_batch.numpy().tolist()
            for i, features in enumerate(features_list):
                yield (features, int(batch_rows[i].label))

# =====================================================
# Hàm chuyển RDD sang DataFrame Spark ML
# =====================================================
def build_feature_dataframe(features_rdd):
    return features_rdd.map(
        lambda x: Row(
            features=Vectors.dense(x[0]),
            label=int(x[1])
        )
    ).toDF()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    INPUT_PATH = "hdfs://namenode:8020/data/ingested/cifake_images"
    OUTPUT_PATH = "hdfs://namenode:8020/data/processed/cifake_features"

    spark = create_spark_session()
    
    print("="*60, file=sys.stderr)
    print("STEP 2: DISTRIBUTED FEATURE EXTRACTION (MobileNetV2 + Batching)", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    # 1. Đọc dữ liệu
    df_images = spark.read.parquet(INPUT_PATH)
    # CHAY FULL 120,000 ANH
    # Repartition để chia tải cho nhiều cores chạy song song
    df_images = df_images.repartition(24) 
    
    print(f"[INFO] Processing {df_images.count()} images (FULL DATASET)...", file=sys.stderr)
    
    # 2. Extract Features
    start_time = time.time()
    
    features_rdd = df_images.rdd.mapPartitions(extract_features_optimized)
    df_features = build_feature_dataframe(features_rdd)
    
    # 3. Write to HDFS
    # Lưu ý: Dùng .write.parquet() sẽ trigger action
    print(f"[INFO] Saving to {OUTPUT_PATH}...", file=sys.stderr)
    df_features.write.mode("overwrite").parquet(OUTPUT_PATH)
    
    # Tính thời gian
    duration = time.time() - start_time
    print("="*60, file=sys.stderr)
    print(f"[DONE] FINISHED in {duration/60:.2f} minutes!", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    spark.stop()
