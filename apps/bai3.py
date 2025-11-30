import org.apache.spark.sql.SparkSession

object DistinctWords {
  def main(args: Array[String]): Unit = {

    // Khởi tạo SparkSession
    val spark = SparkSession.builder()
      .appName("DistinctWordsExample")
      .master("local")
      .getOrCreate()

    // Dữ liệu đầu vào
    val rdd = spark.sparkContext.parallelize(
      List("Germany India USA", "USA India Russia", "India Brazil Canada China")
    )

    // Tách từng từ trong mỗi chuỗi
    val wordsRdd = rdd.flatMap(_.split(" "))

    // Loại bỏ các từ trùng nhau
    val uniqueWords = wordsRdd.distinct()

    // In ra các từ duy nhất
    println("Các từ không trùng:")
    uniqueWords.foreach(println)

    // Dừng Spark
    spark.stop()
  }
}
