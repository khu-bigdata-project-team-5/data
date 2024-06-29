import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf, explode, concat_ws, lit, expr, sort_array
from pyspark.sql.types import StringType, ArrayType
from langid.langid import LanguageIdentifier, model
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
import pandas as pd
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

df = pd.read_csv("Course_info_with_thumbnail.csv")


filtered_course_info_en = df[df['language'] == 'English']
filtered_course_info_en = filtered_course_info_en[['id', 'title', 'headline']]

filtered_course_info_en['title_headline'] = filtered_course_info_en['title'] + " " + filtered_course_info_en['headline']

filtered_course_info_en = filtered_course_info_en.dropna()

spark_df = spark.createDataFrame(filtered_course_info_en)


tokenizer = Tokenizer(inputCol="title_headline", outputCol="words")
tokenized_df = tokenizer.transform(spark_df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_words_df = remover.transform(tokenized_df)

keywords_df = filtered_words_df \
    .withColumn("keyword", explode(col("filtered_words"))) \
    .groupBy("id") \
    .agg(expr("collect_list(keyword)").alias("keywords"))


keywords_df.show(truncate=False)

keywords_pd_df = keywords_df.toPandas()
keywords_pd_df.to_csv("udemy_en.csv", index=False)