import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf, explode, concat_ws, lit, expr, sort_array, array
from pyspark.sql.types import StringType, ArrayType
from langid.langid import LanguageIdentifier, model
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
import pandas as pd
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

merged_course_info = pd.read_csv("udemy_courses.csv")

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

def extract_nouns(text):
    if text is None:
        return []
    
    ko_nouns = [word for word, pos in mecab.pos(text) if pos in ['NNG', 'NP']]
    en_words = [word for word in text.split() if word.isalpha()]
    
    return ko_nouns + en_words

extract_nouns_udf = udf(extract_nouns, ArrayType(StringType()))
filtered_course_info_ko = filtered_course_info_ko.withColumn("title_nouns", extract_nouns_udf(col("title")))
filtered_course_info_ko = filtered_course_info_ko.withColumn("headline_nouns", extract_nouns_udf(col("headline")))

filtered_course_info_ko = filtered_course_info_ko.withColumn("title_nouns_str", concat_ws(", ", col("title_nouns")))
filtered_course_info_ko = filtered_course_info_ko.withColumn("headline_nouns_str", concat_ws(", ", col("headline_nouns")))


filtered_course_info_ko = filtered_course_info_ko.select("id", "title", "title_nouns_str", "headline", "headline_nouns_str")

selected_df = filtered_course_info_ko.withColumn("keywords", concat_ws(", ", col("title_nouns_str"), col("headline_nouns_str")))

selected_df = selected_df.select("id", "keywords")

selected_df = selected_df.dropDuplicates()
selected_df.to_csv("udemy_ko.csv", index=False)