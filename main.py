from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract

spark = SparkSession.builder.appName("IMDBSentimentAnalysis").getOrCreate()

# Read the data as a single column
data = spark.read.option("header", "true").text("movie.csv")


# Use regular expression to extract text and label
data = data.withColumn("text", regexp_extract(data["value"], '^"(.+?)"', 1))
data = data.withColumn("label", regexp_extract(data["value"], '(\d)$', 1))

# Drop the original value column
data = data.drop("value")

# Show the data
# data.show(truncate=False)

data.show()
data.describe().show()

# Show all unique labels
data.select("label").distinct().show()


# Filter and show rows with missing or whitespace-only labels
data.filter(data["label"].isNull() | (data["label"] == "")).show(truncate=False)


# Add a 'line_number' column to the dataset
data_with_line_number = data.withColumn("line_number", monotonically_increasing_id() + 1)

# Filter the rows where both 'text' and 'label' are empty or whitespace-only
filtered_rows = data_with_line_number.filter(
    (data_with_line_number["text"].isNull() | (data_with_line_number["text"] == "")) &
    (data_with_line_number["label"].isNull() | (data_with_line_number["label"] == ""))
)

# Show the line numbers of the filtered rows
filtered_rows.select("line_number").show()
