from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql.functions import lit,collect_set,udf,row_number,arrays_zip,expr,split,collect_list,explode,col,desc
from pyspark.sql.window import Window
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.sql.types import FloatType,StringType,ArrayType, DoubleType,StructType,StructField,BooleanType,DateType,IntegerType
from pyspark.sql.functions import sum as sql_sum

import os
import sys
import pyspark
import codecs

from pyspark.ml.feature import Word2Vec

import numpy as np
from konlpy.tag import Mecab

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
os.environ["SPARK_HOME"] = "/home/hadoop/spark"


mecab = Mecab()
spark = SparkSession.builder.appName("INFoU").getOrCreate()

df = spark.read.format("csv") \
               .option("header", "true") \
               .option("inferSchema", "true") \
               .option("encoding", "UTF-8") \
               .option("multiline", "true") \
               .load("combined_course_reviews_2.csv")

pos_tags = ['NNG', 'MAG', 'VV', 'SL', 'VA']
def tokenize_and_filter(text):
    tokens = [word for word, pos in mecab.pos(text) if pos in pos_tags]
    return tokens

tokenize_and_filter_udf = udf(tokenize_and_filter, ArrayType(StringType()))

grouped_df = df.groupBy("lecture_id").agg(collect_list("comment").alias("comments"))

# Explode comments array into individual comments
exploded_comments_df = grouped_df.withColumn("comment", explode("comments")).select("lecture_id", "comment")

# Tokenize each comment into words
tokenized_df = exploded_comments_df.withColumn("words",tokenize_and_filter_udf(col("comment")))

# Apply HashingTF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
tf = hashingTF.transform(tokenized_df)

# Apply IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf)

def extract_tfidf(row):
    lecture_id = row.lecture_id
    words = row.words
    features = row.features
    indices = features.indices
    values = features.values

    # Ensure we do not go out of bounds
    word_tfidf_pairs = []
    for i in range(len(indices)):
        if i < len(words):
            word_tfidf_pairs.append((lecture_id, words[i],float(values[i])))
    return word_tfidf_pairs


tfidf_rdd = tfidf.rdd.flatMap(extract_tfidf)

# Convert to DataFrame
tfidf_df = spark.createDataFrame(tfidf_rdd, ["lecture_id", "word", "tfidf_value"])
tfidf_df = tfidf_df.dropDuplicates(['lecture_id', 'word'])

tfidf_df = tfidf_df.dropna(subset=['lecture_id'])
tfidf_df = tfidf_df.orderBy(['lecture_id', 'tfidf_value'], ascending=[True, False])


# 최종 난이도 점수 계산 및 정렬

df_with_final_scores = tfidf_df.groupBy("lecture_id")
    
# 결과 확인
df_with_final_scores.show(truncate=False)