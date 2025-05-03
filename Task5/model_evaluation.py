from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark
spark = SparkSession.builder.appName("Task5-ModelEvaluation").getOrCreate()

# Load prediction output from Task 4
predictions = spark.read.csv("/workspaces/assignment-3-fake-news-detection-ramala-prasanth/Task4/task4_output/part-00000-1bb8797d-f355-45c4-bc3f-a8bd5ff4e408-c000.csv", header=True, inferSchema=True)

# Cast necessary columns
predictions = predictions.withColumn("label_index", predictions["label_index"].cast("double")) \
                         .withColumn("prediction", predictions["prediction"].cast("double"))

# Initialize evaluators
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy")

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1")

# Evaluate
accuracy = evaluator_accuracy.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

# Print in markdown table format
print("| Metric    | Value |")
print("|-----------|-------|")
print(f"| Accuracy  | {accuracy:.2f} |")
print(f"| F1 Score  | {f1_score:.2f} |")

# Save to CSV
results = spark.createDataFrame([
    ("Accuracy", round(accuracy, 4)),
    ("F1 Score", round(f1_score, 4))
], ["Metric", "Value"])

results.write.csv("task5_output.csv", header=True, mode="overwrite")
