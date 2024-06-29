import os
import numpy as np
from pyspark.ml.linalg import SparseVector
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf, concat_ws, lit
from pyspark.sql.types import StringType, ArrayType
from langid.langid import LanguageIdentifier, model
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from konlpy.tag import Mecab

spark = SparkSession.builder.appName("UdemyTitleAnalysis").config("spark.sql.autoBroadcastJoinThreshold", "-1").getOrCreate()

merged_course_info = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("encoding", "UTF-8") \
    .load("udemy_courses.csv")

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
filtered_course_info_ko = filtered_course_info.filter(col("language") == "ko")

mecab = Mecab()
pos_tags = ['NNG', 'MAG', 'VV', 'SL', 'VA']

def tokenize_and_filter_ko(text):
    if text is None:
        return []
    tokens = [word for word, pos in mecab.pos(text) if pos in pos_tags]
    return tokens

tokenize_and_filter_ko_udf = udf(tokenize_and_filter_ko, ArrayType(StringType()))

filtered_course_info_ko = filtered_course_info_ko.withColumn("title_words", tokenize_and_filter_ko_udf(col("title")))
filtered_course_info_ko = filtered_course_info_ko.withColumn("headline_words", tokenize_and_filter_ko_udf(col("headline")))

filtered_course_info_ko = filtered_course_info_ko.withColumn("title_word", concat_ws(", ", "title_words"))
filtered_course_info_ko = filtered_course_info_ko.withColumn("headline_word", concat_ws(", ", "headline_words"))

filtered_course_info_ko = filtered_course_info_ko.dropDuplicates()

# TF-IDF
tokenizer = Tokenizer(inputCol="title_word", outputCol="title_words_token")
filtered_course_info_ko = tokenizer.transform(filtered_course_info_ko)

tokenizer = Tokenizer(inputCol="headline_word", outputCol="headline_words_token")
filtered_course_info_ko = tokenizer.transform(filtered_course_info_ko)

# CountVectorizer
cv_title = CountVectorizer(inputCol="title_words_token", outputCol="raw_title_features", vocabSize=100)
cv_model_title = cv_title.fit(filtered_course_info_ko)
featurizedData_title = cv_model_title.transform(filtered_course_info_ko)

cv_headline = CountVectorizer(inputCol="headline_words_token", outputCol="raw_headline_features", vocabSize=100)
cv_model_headline = cv_headline.fit(filtered_course_info_ko)
featurizedData_headline = cv_model_headline.transform(filtered_course_info_ko)

# IDF
idf_title = IDF(inputCol="raw_title_features", outputCol="title_tfidf")
idfModel_title = idf_title.fit(featurizedData_title)
rescaledData_title = idfModel_title.transform(featurizedData_title)

idf_headline = IDF(inputCol="raw_headline_features", outputCol="headline_tfidf")
idfModel_headline = idf_headline.fit(featurizedData_headline)
rescaledData_headline = idfModel_headline.transform(featurizedData_headline)

rescaledData_ko = rescaledData_title.join(rescaledData_headline, ["id"])

vocabulary_title = cv_model_title.vocabulary
vocabulary_headline = cv_model_headline.vocabulary

def extract_top_keywords(tfidf_values, vocabulary, threshold=1.5):
    if tfidf_values is None:
        return []
    
    if isinstance(tfidf_values, SparseVector):
        tfidf_values = tfidf_values.toArray()
    
    sorted_keywords = sorted(enumerate(tfidf_values), key=lambda x: x[1], reverse=True)
    top_keywords = [vocabulary[index] for index, value in sorted_keywords if value >= threshold]
    return top_keywords

extract_top_title_keywords_udf = udf(lambda x: extract_top_keywords(x, vocabulary_title), ArrayType(StringType()))
extract_top_headline_keywords_udf = udf(lambda x: extract_top_keywords(x, vocabulary_headline), ArrayType(StringType()))

rescaledData_ko = rescaledData_ko.withColumn("title_keywords", extract_top_title_keywords_udf(col("title_tfidf")))
rescaledData_ko = rescaledData_ko.withColumn("headline_keywords", extract_top_headline_keywords_udf(col("headline_tfidf")))


rescaledData_ko = rescaledData_ko.withColumn("title_keywords", concat_ws(", ", "title_keywords"))
rescaledData_ko = rescaledData_ko.withColumn("headline_keywords", concat_ws(", ", "headline_keywords"))

selected_columns = ['id', 'title_keywords', 'headline_keywords']
udemy_keyword_ko = rescaledData_ko[selected_columns]


csv_filename = "udemy_keyword_ko.csv"
udemy_keyword_ko.write.csv(csv_filename, header=True, mode="overwrite")


os.system("hadoop fs -put udemy_keyword_ko.csv /user/hadoop/")
