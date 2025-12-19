from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import sys
import time

# Khoi tao Spark Session
def create_spark_session():
    spark = SparkSession.builder \
        .appName("CIFAKE-Step-3-Classification") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# Ham train va danh gia model
def train_and_evaluate(spark, features_path, model_output_path):
    
    # Doc du lieu features tu HDFS
    print("[INFO] Loading features from HDFS...", file=sys.stderr)
    df = spark.read.parquet(features_path)
    total_count = df.count()
    print(f"[INFO] Total samples: {total_count}", file=sys.stderr)
    
    # Chia du lieu theo yeu cau: 83.3% train, 16.7% test
    print("[INFO] Splitting data: 83.3% train, 16.7% test...", file=sys.stderr)
    train_data, test_data = df.randomSplit([0.833, 0.167], seed=42)
    train_count = train_data.count()
    test_count = test_data.count()
    print(f"[INFO] Train samples: {train_count}", file=sys.stderr)
    print(f"[INFO] Test samples: {test_count}", file=sys.stderr)
    
    # Cache du lieu de tang toc
    train_data.cache()
    test_data.cache()
    
    # Khoi tao model Logistic Regression
    print("[INFO] Setting up Logistic Regression with CrossValidator...", file=sys.stderr)
    lr = LogisticRegression(
        labelCol="label",
        featuresCol="features"
    )
    
    # Tao luoi tham so de tim gia tri tot nhat (Hyperparameter Tuning)
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.maxIter, [10, 20]) \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .build()
    
    # Evaluator de danh gia trong qua trinh tuning
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    # CrossValidator: 3-fold cross validation
    crossval = CrossValidator(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4  # Chay song song 4 model cung luc
    )
    
    # Train model voi CrossValidation
    print("[INFO] Training with 3-fold Cross Validation...", file=sys.stderr)
    start_time = time.time()
    cv_model = crossval.fit(train_data)
    train_time = time.time() - start_time
    print(f"[INFO] Training completed in {train_time:.2f} seconds", file=sys.stderr)
    
    # Lay model tot nhat
    best_model = cv_model.bestModel
    print(f"[INFO] Best model params: maxIter={best_model.getMaxIter()}, regParam={best_model.getRegParam()}", file=sys.stderr)
    
    # Du doan tren tap test
    print("[INFO] Evaluating on test set...", file=sys.stderr)
    predictions = best_model.transform(test_data)
    
    # Tinh cac chi so danh gia
    accuracy = evaluator.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    f1_score = evaluator_f1.evaluate(predictions)
    
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    precision = evaluator_precision.evaluate(predictions)
    
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )
    recall = evaluator_recall.evaluate(predictions)
    
    # In ket qua
    print("=" * 50, file=sys.stderr)
    print("CLASSIFICATION RESULTS (on 20k test set)", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Accuracy:  {accuracy * 100:.2f}%", file=sys.stderr)
    print(f"F1 Score:  {f1_score * 100:.2f}%", file=sys.stderr)
    print(f"Precision: {precision * 100:.2f}%", file=sys.stderr)
    print(f"Recall:    {recall * 100:.2f}%", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    
    # Confusion Matrix
    print("[INFO] Confusion Matrix:", file=sys.stderr)
    predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()
    
    # Luu model tot nhat ra HDFS
    print(f"[INFO] Saving best model to {model_output_path}...", file=sys.stderr)
    best_model.write().overwrite().save(model_output_path)
    print("[DONE] Model saved successfully!", file=sys.stderr)
    
    return accuracy, f1_score

# Main
if __name__ == "__main__":
    # Duong dan input/output
    FEATURES_PATH = "hdfs://namenode:8020/data/processed/cifake_features"
    MODEL_OUTPUT_PATH = "hdfs://namenode:8020/models/cifake_classifier"
    
    # Khoi tao Spark
    spark = create_spark_session()
    
    print("=" * 50, file=sys.stderr)
    print("STEP 3: DISTRIBUTED CLASSIFICATION", file=sys.stderr)
    print("Train: 80% samples, Test: 20% samples", file=sys.stderr)
    print("Method: Logistic Regression + CrossValidation", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    
    # Train va danh gia
    accuracy, f1 = train_and_evaluate(spark, FEATURES_PATH, MODEL_OUTPUT_PATH)
    
    print("=" * 50, file=sys.stderr)
    print(f"[DONE] Step 3 completed! Final Accuracy: {accuracy*100:.2f}%", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    
    spark.stop()
