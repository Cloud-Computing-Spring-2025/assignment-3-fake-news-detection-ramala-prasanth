from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import SparseVector, VectorUDT
import re

# Start Spark session
spark = SparkSession.builder.appName("Fake News Task 4").getOrCreate()

# Load transformed data (from Task 3)
df = spark.read.csv(
    "/workspaces/assignment-3-fake-news-detection-ramala-prasanth/Task3/task3_output/part-00000-6019f872-2b99-4683-b8ac-7551e1b62221-c000.csv", 
    header=True, 
    inferSchema=True
)

# Load original data to get 'title' (for final output)
original_df = spark.read.csv(
    "/workspaces/assignment-3-fake-news-detection-ramala-prasanth/Task2/task2_output/part-00000-846287ca-a47f-4c40-a268-99cb671ff083-c000.csv", 
    header=True, 
    inferSchema=True
)

# Rename 'title' in original_df to avoid ambiguity
original_df = original_df.withColumnRenamed("title", "original_title")

# UDF to parse sparse vector from string
def parse_sparse_vector(s):
    match = re.match(r"\((\d+),\[(.*?)\],\[(.*?)\]\)", s)
    if not match:
        return SparseVector(0, [])
    size = int(match.group(1))
    indices = list(map(int, match.group(2).split(","))) if match.group(2) else []
    values = list(map(float, match.group(3).split(","))) if match.group(3) else []
    return SparseVector(size, indices, values)

vector_udf = udf(parse_sparse_vector, VectorUDT())

# Convert features column from string to SparseVector
df = df.withColumn("features", vector_udf(col("features")))

# Split into training and test data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)

# Join test data with original_df to get titles
test_data = test_data.join(original_df, on="id", how="left")

# Predict on test data
predictions = model.transform(test_data)

# Select required columns and rename for clarity
predictions.select("id", "original_title", "label_index", "prediction") \
    .withColumnRenamed("original_title", "title") \
    .write.csv("task4_output", header=True, mode="overwrite")
