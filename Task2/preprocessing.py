from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Step 1: Start Spark session
spark = SparkSession.builder.appName("Fake News Task 2").getOrCreate()

# Step 2: Read input CSV
df = spark.read.csv("/workspaces/assignment-3-fake-news-detection-ramala-prasanth/Task1/task1_output.csv/part-00000-2519d8a7-9525-4e73-97b7-5de3ef303595-c000.csv", header=True, inferSchema=True)

# Step 3: Convert text column to lowercase
df_lower = df.withColumn("text", lower(col("text")))

# Step 4: Tokenize text into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df_tokenized = tokenizer.transform(df_lower)

# Step 5: Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_cleaned = remover.transform(df_tokenized)

# Optional: Create a temporary view
df_cleaned.createOrReplaceTempView("cleaned_news")

# Step 6: Prepare final DataFrame (convert array to space-separated string)
df_output = df_cleaned.select(
    "id",
    "title",
    concat_ws(" ", "filtered_words").alias("filtered_words"),
    "label"
)

# Step 7: Write output to CSV
df_output.write.csv("task2_output", header=True, mode="overwrite")
