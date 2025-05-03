from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.sql.functions import col, split, regexp_replace

# Initialize Spark
spark = SparkSession.builder.appName("Task3-FeatureExtraction").getOrCreate()

# Load preprocessed data from Task 2
df = spark.read.csv("/workspaces/assignment-3-fake-news-detection-ramala-prasanth/Task2/task2_output/part-00000-846287ca-a47f-4c40-a268-99cb671ff083-c000.csv", header=True, inferSchema=True)

# Convert filtered_words string to array
df = df.withColumn("filtered_words", regexp_replace("filtered_words", r"[\[\]\']", "")) \
       .withColumn("filtered_words", split(col("filtered_words"), ", "))

# TF-IDF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurizedData = hashingTF.transform(df)

idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Label indexing
indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed = indexer.fit(rescaledData).transform(rescaledData)

# Convert array and vector columns to string for CSV output
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

vector_to_string = udf(lambda v: str(v), StringType())
array_to_string = udf(lambda arr: ", ".join(arr), StringType())

output_df = indexed.withColumn("features_str", vector_to_string(col("features"))) \
                   .withColumn("filtered_words_str", array_to_string(col("filtered_words")))

# Select and rename columns for output
final_df = output_df.select(
    col("id"),
    col("title"),
    col("filtered_words_str").alias("filtered_words"),
    col("features_str").alias("features"),
    col("label_index")
)

# Save as CSV
final_df.write.csv("task3_output", header=True, mode="overwrite")
