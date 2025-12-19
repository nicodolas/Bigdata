from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, IntegerType
import sys

def extract_images_from_zip_distributed(spark, zip_path, output_dir):
    """
    Step 1: Extract ZIP file from HDFS và distribute việc xử lý
    Approach: 
    1. Load ZIP từ HDFS
    2. Extract list of files
    3. Parallelize list và distribute việc đọc từng ảnh
    """
    import zipfile
    import io
    
    print("="*70, file=sys.stderr)
    print("STEP 1: DATA INGESTION", file=sys.stderr)
    print("="*70, file=sys.stderr)
    
    # Step 1.1: Load ZIP file từ HDFS
    print(f"📂 Reading ZIP from HDFS: {zip_path}", file=sys.stderr)
    df_zip = spark.read.format("binaryFile").load(zip_path)
    zip_row = df_zip.collect()[0]
    zip_bytes = zip_row.content
    
    print(f"📦 ZIP size: {len(zip_bytes) / (1024**2):.2f} MB", file=sys.stderr)
    
    # Step 1.2: Extract file list từ ZIP
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Filter chỉ lấy .jpg files
        file_list = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📋 Found {len(file_list)} image files in ZIP", file=sys.stderr)
    
    # Step 1.3: Tạo list of (filename, label) để parallelize
    file_info_list = []
    for file_name in file_list:
        # Determine label
        if "/REAL/" in file_name.upper() or "\\REAL\\" in file_name.upper():
            label = 1
        elif "/FAKE/" in file_name.upper() or "\\FAKE\\" in file_name.upper():
            label = 0
        else:
            continue
        
        file_info_list.append((file_name, label))
    
    print(f"✅ Valid labeled images: {len(file_info_list)}", file=sys.stderr)
    
    # Step 1.4: Parallelize file list - DISTRIBUTED!
    # Mỗi partition sẽ xử lý một phần files
    print(f"🔄 Parallelizing extraction across Spark cluster...", file=sys.stderr)
    
    # Broadcast ZIP bytes để workers có thể access
    zip_bytes_bc = spark.sparkContext.broadcast(zip_bytes)
    
    # Create RDD from file list
    files_rdd = spark.sparkContext.parallelize(file_info_list, numSlices=8)
    
    # Define UDF to extract images in parallel
    def extract_image_from_zip(file_info_batch):
        """Extract images - chạy trong mỗi partition (distributed)"""
        import zipfile
        import io
        
        zip_data = zip_bytes_bc.value
        
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            for file_name, label in file_info_batch:
                try:
                    # Read image content
                    img_bytes = z.read(file_name)
                    
                    # Clean filename
                    clean_name = file_name.replace('\\', '/').split('/')[-1]
                    
                    # Yield tuple (not list!)
                    yield (clean_name, img_bytes, label)
                    
                except Exception as e:
                    print(f"Warning: Cannot read {file_name}: {e}", file=sys.stderr)
                    continue
    
    # Extract images distributed across partitions
    images_rdd = files_rdd.mapPartitions(extract_image_from_zip)
    
    # Convert to DataFrame
    schema = StructType([
        StructField("filename", StringType(), False),
        StructField("content", BinaryType(), False),
        StructField("label", IntegerType(), False)
    ])
    
    df_images = spark.createDataFrame(images_rdd, schema)
    
    # Count before writing
    num_images = df_images.count()
    print(f"📊 Extracted {num_images} images successfully", file=sys.stderr)
    
    # Step 1.5: Write to HDFS as Parquet
    print(f"💾 Writing to HDFS: {output_dir}", file=sys.stderr)
    df_images.write.mode("overwrite").parquet(output_dir)
    
    print("="*70, file=sys.stderr)
    print(f"✅ STEP 1 COMPLETED: {num_images} images ingested to HDFS", file=sys.stderr)
    print("="*70, file=sys.stderr)
    
    # Clean up broadcast variable
    zip_bytes_bc.unpersist()
    
    return num_images


if __name__ == "__main__":
    # Khởi tạo Spark
    spark = SparkSession.builder \
        .appName("CIFAKE-Step-1-Ingestion-Distributed") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Paths
    ZIP_PATH = "hdfs://namenode:8020/data/raw/dataset.zip"
    OUTPUT_DIR = "hdfs://namenode:8020/data/ingested/cifake_images"
    
    # Run ingestion
    num_images = extract_images_from_zip_distributed(spark, ZIP_PATH, OUTPUT_DIR)
    
    print(f"\n🎉 Ingestion successful: {num_images} images ready for processing!", file=sys.stderr)
    
    spark.stop()
