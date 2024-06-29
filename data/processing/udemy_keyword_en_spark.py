import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf, explode, concat_ws, lit, expr, sort_array
from pyspark.sql.types import StringType, ArrayType
from langid.langid import LanguageIdentifier, model
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF

spark = SparkSession.builder.appName("UdemyTitleAnalysis").config("spark.sql.autoBroadcastJoinThreshold", "-1").getOrCreate()

merged_course_info = spark.read.format("csv") \
                       .option("header", "true") \
                       .option("inferSchema", "true") \
                       .option("encoding", "UTF-8") \
                       .load("Course_info_with_thumbnail.csv")

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
def detect_language_langid(text):
    try:
        lang, _ = identifier.classify(text)
        return lang
    except:
        return "Unknown"

detect_language_udf = udf(detect_language_langid, StringType())

merged_course_info = merged_course_info.withColumn("language", detect_language_udf(col("title")))

filtered_course_info = merged_course_info.filter((col("language") == "en") | (col("language") == "ko"))
filtered_course_info_en = filtered_course_info.filter(col("language") == "en")

filtered_course_info_en = filtered_course_info_en.select("id", "title", "headline") \
                                                 .withColumn("title_lower", lower(col("title"))) \
                                                 .withColumn("headline_lower", lower(col("headline"))) \
                                                 .withColumn("title_headline", concat_ws(" ", col("title_lower"), col("headline_lower")))

filtered_course_info_en = filtered_course_info_en.na.drop()

tokenizer = Tokenizer(inputCol="title_headline", outputCol="words")
filtered_course_info_en = tokenizer.transform(filtered_course_info_en)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_course_info_en = remover.transform(filtered_course_info_en)

keywords = filtered_course_info_en \
    .withColumn("keyword", explode(col("filtered_words"))) \
    .groupBy("id") \
    .agg(expr("collect_list(keyword)").alias("keywords"))

filtered_course_info_en = filtered_course_info_en.join(keywords, "id", "left")

filtered_course_info_en = filtered_course_info_en.drop("title_lower", "headline_lower", "words", "filtered_words")
filtered_course_info_en = filtered_course_info_en.dropna(subset=["keywords"]).dropDuplicates()



filtered_course_info_en = filtered_course_info_en.withColumn("keywords_str", concat_ws(", ", "keywords"))

final_df = filtered_course_info_en.select("id", "title", "headline", "title_headline", "keywords_str")

merged_df = final_df
merged_df = pd.concat(all_dataframes, ignore_index=True)
merged_df.drop_duplicates(inplace=True)

merged_df.dropna(inplace=True)
merged_df = merged_df[["id", "keywords_str"]]


merged_df['keywords_array'] = merged_df['keywords_str'].str.split(',')
merged_df = merged_df[["id", "keywords_array"]]

merged_df['keywords_array'] = merged_df['keywords_array'].apply(lambda x: list(dict.fromkeys(x)))

pd.set_option('display.max_colwidth', None)

merged_df.to_csv('udemy_en.csv', index=False)