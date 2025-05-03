from pyspark.sql import SparkSession

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Fake News Detection - Task 1") \
    .getOrCreate()

# Step 2: Load the CSV and Infer Schema
df = spark.read.csv("/workspaces/assignment-3-fake-news-detection-ramala-prasanth/fake_news_sample.csv", header=True, inferSchema=True)

# Step 3: Create a Temporary View
df.createOrReplaceTempView("news_data")

# Step 4: Show First 5 Rows
print("First 5 Rows:")
df.show(5, truncate=False)

# Step 5: Count Total Number of Articles
print("Total Number of Articles:")
spark.sql("SELECT COUNT(*) AS total_articles FROM news_data").show()

# Step 6: Retrieve Distinct Labels
print("Distinct Labels:")
spark.sql("SELECT DISTINCT label FROM news_data").show()

# Step 7: Write Output to CSV (Sample Output)
sample_output = df  # or use any query like filtering by label
sample_output.write.csv("task1_output.csv", header=True, mode="overwrite")
