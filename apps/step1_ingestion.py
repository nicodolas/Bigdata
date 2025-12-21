from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, IntegerType
import sys

def extract_images_from_zip_distributed(spark, zip_path, output_dir):
    """
    Bước 1: Giải nén tệp ZIP từ HDFS và phân phối quá trình xử lý
    Cách tiếp cận: 
    1. Tải tệp ZIP từ HDFS
    2. Trích xuất danh sách các tệp
    3. Song song hóa danh sách và phân phối việc đọc từng hình ảnh
    """
    import zipfile
    import io
    
    print("="*70, file=sys.stderr)
    print("BƯỚC 1: THU THẬP DỮ LIỆU", file=sys.stderr)
    print("="*70, file=sys.stderr)
    
    # Bước 1.1: Tải tệp ZIP từ HDFS
    print(f"Đang đọc file ZIP từ HDFS: {zip_path}", file=sys.stderr)
    df_zip = spark.read.format("binaryFile").load(zip_path)
    zip_row = df_zip.collect()[0]
    zip_bytes = zip_row.content
    
    print(f"Kích thước file ZIP: {len(zip_bytes) / (1024**2):.2f} MB", file=sys.stderr)
    
    # Bước 1.2: Trích xuất danh sách tệp từ ZIP
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Chỉ lấy các tệp hình ảnh
        file_list = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Tìm thấy {len(file_list)} tệp hình ảnh trong ZIP", file=sys.stderr)
    
    # Bước 1.3: Tạo danh sách (tên tệp, nhãn) để song song hóa
    file_info_list = []
    for file_name in file_list:
        # Xác định nhãn
        if "/REAL/" in file_name.upper() or "\\REAL\\" in file_name.upper():
            label = 1
        elif "/FAKE/" in file_name.upper() or "\\FAKE\\" in file_name.upper():
            label = 0
        else:
            continue
        
        file_info_list.append((file_name, label))
    
    print(f"Số lượng ảnh hợp lệ có nhãn: {len(file_info_list)}", file=sys.stderr)
    
    # Bước 1.4: Song song hóa danh sách tệp - PHÂN PHỐI!
    # Mỗi phân vùng sẽ xử lý một phần tệp
    print(f"Đang thực hiện phân phối việc giải nén trên cụm Spark...", file=sys.stderr)
    
    # Broadcast nội dung file ZIP để các workers có thể truy cập
    zip_bytes_bc = spark.sparkContext.broadcast(zip_bytes)
    
    # Tạo RDD từ danh sách tệp
    files_rdd = spark.sparkContext.parallelize(file_info_list, numSlices=8)
    
    # Định nghĩa hàm UDF để giải nén ảnh song song
    def extract_image_from_zip(file_info_batch):
        """Giải nén hình ảnh - chạy trong mỗi phân vùng (phân phối)"""
        import zipfile
        import io
        
        zip_data = zip_bytes_bc.value
        
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            for file_name, label in file_info_batch:
                try:
                    # Đọc nội dung ảnh
                    img_bytes = z.read(file_name)
                    
                    # Làm sạch tên tệp
                    clean_name = file_name.replace('\\', '/').split('/')[-1]
                    
                    # Trả về bộ dữ liệu (tuple)
                    yield (clean_name, img_bytes, label)
                    
                except Exception as e:
                    print(f"Cảnh báo: Không thể đọc {file_name}: {e}", file=sys.stderr)
                    continue
    
    # Giải nén hình ảnh phân phối trên các phân vùng
    images_rdd = files_rdd.mapPartitions(extract_image_from_zip)
    
    # Chuyển đổi thành DataFrame
    schema = StructType([
        StructField("filename", StringType(), False),
        StructField("content", BinaryType(), False),
        StructField("label", IntegerType(), False)
    ])
    
    df_images = spark.createDataFrame(images_rdd, schema)
    
    # Đếm số lượng trước khi ghi
    num_images = df_images.count()
    print(f"Đã trích xuất thành công {num_images} hình ảnh", file=sys.stderr)
    
    # Bước 1.5: Ghi vào HDFS dưới dạng Parquet
    print(f"Đang ghi kết quả vào HDFS: {output_dir}", file=sys.stderr)
    df_images.write.mode("overwrite").parquet(output_dir)
    
    print("="*70, file=sys.stderr)
    print(f"BƯỚC 1 HOÀN TẤT: {num_images} hình ảnh đã được nạp vào HDFS", file=sys.stderr)
    print("="*70, file=sys.stderr)
    
    # Dọn dẹp biến broadcast
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
    
    # Đường dẫn
    ZIP_PATH = "hdfs://namenode:8020/data/raw/dataset.zip"
    OUTPUT_DIR = "hdfs://namenode:8020/data/ingested/cifake_images"
    
    # Chạy quá trình nạp dữ liệu
    num_images = extract_images_from_zip_distributed(spark, ZIP_PATH, OUTPUT_DIR)
    
    print(f"\nNạp dữ liệu thành công: {num_images} hình ảnh đã sẵn sàng để xử lý!", file=sys.stderr)
    
    spark.stop()
