from pyspark.sql.functions import lit,collect_set,udf,row_number,arrays_zip,expr,split,collect_list,explode,col,desc,when
from pyspark.sql.functions import min, max, mean,count
from pyspark.sql.types import FloatType,StringType,ArrayType, DoubleType,StructType,StructField,BooleanType,DateType,IntegerType
from pyspark.sql.functions import sum as sql_sum
import os

from pyspark.ml.feature import Word2Vec

import numpy as np
from konlpy.tag import Mecab
from pyspark.sql import SparkSession

# os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
os.environ["SPARK_HOME"] = "/home/hadoop/spark"


mecab = Mecab()
# spark = SparkSession.builder \
#     .appName("YourAppName") \
#     .config("spark.driver.extraJavaOptions", "--add-opens java.base/java.net=ALL-UNNAMED") \
#     .config("spark.executor.extraJavaOptions", "--add-opens java.base/java.net=ALL-UNNAMED") \
#     .getOrCreate()
spark = SparkSession.builder.appName("INFoU").getOrCreate()
# start_time = time.time()

df = spark.read.format("csv") \
               .option("header", "true") \
               .option("inferSchema", "true") \
               .option("encoding", "UTF-8") \
               .option("multiline", "true") \
               .load("combined_course_reviews_2.csv")

df2 = spark.read.format("csv") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .option("encoding", "UTF-8") \
                .option("multiline", "true") \
                .load("inflearn_course_info.csv").select("course_identification", "star_rate")
# 파일 읽기 종료 시간 기록
# end_time = time.time()

# 파일 읽기에 소요된 시간 계산
# print("파일 읽기 시간:", end_time - start_time, "초")
# start_time = time.time()
pos_tags = ['NNG', 'MAG', 'VV', 'SL', 'VA']
def tokenize_and_filter(text):
    if text is None:
        return []
    tokens = [word for word, pos in mecab.pos(text) if pos in pos_tags]
    return tokens

tokenize_and_filter_udf = udf(tokenize_and_filter, ArrayType(StringType()))

tokenized_df = df.withColumn("words",tokenize_and_filter_udf(col("comment")))

word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="words", outputCol="result")
model = word2Vec.fit(tokenized_df)

word_vectors = {row['word']: row['vector'].toArray() for row in model.getVectors().collect()}

def get_word_vector(word):
    return word_vectors.get(word, np.zeros(10))

# 난이도 키워드
difficulty_pos_keywords = ["초보","쉬움","이지","쉽","이해","잘","좋","추천","기초"]
difficulty_neg_keywords = ["어렵","하드","어려움","안","부족"]

# 강의력 키워드
lecture_pos_keywords = ["이해","추천", "자료","내용","근본","꼼꼼","발표","비유"]
lecture_neg_keywords = ["대충","부족","어려움"]

#강의자료 키워드
data_pos_keywords = ["비유","소개","명시","이미지","애니메이션","문서","코드","내용","방법","과정","방식","검색"]
data_neg_keywords = ["부실","부족","어려움"]

#실습자료 키워드
train_pos_keywords = ["코드","예제","방법","이미지","프로젝트","개발"]
train_neg_keywords = ["부실","부족","어려움"]
# 프로젝트, 개발, 비유, 연습, 연구, 문서 , 코드 , 검색, 내용, 방법, 소개, 명시, 개념, 과정, 방식, 이미지, 애니메이션


# match_keywords_udf = udf(lambda words: match_keywords(words, pos_vectors, neg_vectors, model), ArrayType(FloatType()))
# def compute_similarity(word, target_words):
#     try:
#         vec = model.transform(spark.createDataFrame([(word,)], ["word"]).select("word")).collect()[0][1]
#         synonyms = model.findSynonyms(vec, len(target_words)).collect()
#         similarity_score = sum(syn[1] for syn in synonyms if syn[0] in target_words)
#         return similarity_score
#     except:
#         return 0.0

# def compute_similarity(word):
#     try:
#         vec = word_vectors.get(word, np.zeros(100))  # Get the vector for the word
#         synonyms = model.findSynonyms(vec, len(word)).collect()  # Find synonyms using the vector
#         similarity_score = sum(len(syn[0]) for syn in synonyms if syn[0] in word)
#         return similarity_score
#     except:
#         return 0.0

def compute_similarity(word):
    try:
        # vec = word_vectors.get(word, np.zeros(10))  # Get the vector for the word
        synonyms = model.findSynonyms(word, len(word)).collect()  # Find synonyms using the vector
        similarity_score = sum(syn[1] for syn in synonyms if syn[1] > 0)
        return similarity_score
    except:
        return 0.0

# Precompute similarity scores for positive and negative keywords
def precompute_keyword_similarities(keywords):
    return {keyword: compute_similarity(keyword) for keyword in keywords}

difficulty_pos_similarities = precompute_keyword_similarities(difficulty_pos_keywords)
difficulty_neg_similarities = precompute_keyword_similarities(difficulty_neg_keywords)

lecture_pos_similarities = precompute_keyword_similarities(lecture_pos_keywords)
lecture_neg_similarities = precompute_keyword_similarities(lecture_neg_keywords)

data_pos_similarities = precompute_keyword_similarities(data_pos_keywords)
data_neg_similarities = precompute_keyword_similarities(data_neg_keywords)

train_pos_similarities = precompute_keyword_similarities(train_pos_keywords)
train_neg_similarities = precompute_keyword_similarities(train_neg_keywords)


def match_keywords(words, difficulty_pos_similarities, difficulty_neg_similarities, lecture_pos_similarities, lecture_neg_similarities, data_pos_similarities, data_neg_similarities, train_pos_similarities, train_neg_similarities):
    diff_pos_score = 0.0
    diff_neg_score = 0.0
    lec_pos_score = 0.0
    lec_neg_score = 0.0
    data_pos_score = 0.0
    data_neg_score = 0.0
    train_pos_score = 0.0
    train_neg_score = 0.0
    for word in words:
        diff_pos_score += difficulty_pos_similarities.get(word, 0.0)
        diff_neg_score += difficulty_neg_similarities.get(word, 0.0)
        lec_pos_score += lecture_pos_similarities.get(word, 0.0)
        lec_neg_score += lecture_neg_similarities.get(word, 0.0)
        data_pos_score += data_pos_similarities.get(word, 0.0)
        data_neg_score += data_neg_similarities.get(word, 0.0)
        train_pos_score += train_pos_similarities.get(word, 0.0)
        train_neg_score += train_neg_similarities.get(word, 0.0)

    pos_score = diff_pos_score + lec_pos_score + data_pos_score + train_pos_score
    neg_score = diff_neg_score + lec_neg_score + data_neg_score + train_neg_score
    
    analysis = 1.0 if pos_score >= neg_score else -1.0
    return diff_pos_score, diff_neg_score, lec_pos_score, lec_neg_score, data_pos_score, data_neg_score, train_pos_score, train_neg_score, analysis

# Register UDF for matching keywords and calculating scores
match_keywords_udf = udf(lambda words: match_keywords(words, difficulty_pos_similarities, difficulty_neg_similarities, lecture_pos_similarities, lecture_neg_similarities, data_pos_similarities, data_neg_similarities, train_pos_similarities, train_neg_similarities), ArrayType(FloatType()))
# 유사도 분석
# end_time = time.time()
# print("형태소분석 및 유사도 분석시간:", end_time - start_time, "초")
# start_time = time.time()
result_df = tokenized_df.withColumn("scores", match_keywords_udf(col("words"))) \
                        .withColumn("diff_pos_score", col("scores").getItem(0)) \
                        .withColumn("diff_neg_score", col("scores").getItem(1)) \
                        .withColumn("lec_pos_score", col("scores").getItem(2)) \
                        .withColumn("lec_neg_score", col("scores").getItem(3)) \
                        .withColumn("data_pos_score", col("scores").getItem(4)) \
                        .withColumn("data_neg_score", col("scores").getItem(5)) \
                        .withColumn("train_pos_score", col("scores").getItem(6)) \
                        .withColumn("train_neg_score", col("scores").getItem(7)) \
                        .withColumn("analysis", col("scores").getItem(8)) \
                        .drop("scores") \
                        .drop("comment")



cnt_condition = lambda cond: sum(when(cond,1).otherwise(0))
grouped_df = result_df.groupBy("lecture_id").agg(
    min("diff_pos_score").alias("min_diff_pos_score"),
    max("diff_pos_score").alias("max_diff_pos_score"),
    mean("diff_pos_score").alias("mean_diff_pos_score"),
    min("diff_neg_score").alias("min_diff_neg_score"),
    max("diff_neg_score").alias("max_diff_neg_score"),
    mean("diff_neg_score").alias("mean_diff_neg_score"),
    min("lec_pos_score").alias("min_lec_pos_score"),
    max("lec_pos_score").alias("max_lec_pos_score"),
    mean("lec_pos_score").alias("mean_lec_pos_score"),
    min("lec_neg_score").alias("min_lec_neg_score"),
    max("lec_neg_score").alias("max_lec_neg_score"),
    mean("lec_neg_score").alias("mean_lec_neg_score"),
    min("data_pos_score").alias("min_data_pos_score"),
    max("data_pos_score").alias("max_data_pos_score"),
    mean("data_pos_score").alias("mean_data_pos_score"),
    min("data_neg_score").alias("min_data_neg_score"),
    max("data_neg_score").alias("max_data_neg_score"),
    mean("data_neg_score").alias("mean_data_neg_score"),
    min("train_pos_score").alias("min_train_pos_score"),
    max("train_pos_score").alias("max_train_pos_score"),
    mean("train_pos_score").alias("mean_train_pos_score"),
    min("train_neg_score").alias("min_train_neg_score"),
    max("train_neg_score").alias("max_train_neg_score"),
    mean("train_neg_score").alias("mean_train_neg_score"),
    count(when(col("analysis") == 1.0, True)).alias("good"),
    count(when(col("analysis") == -1.0, True)).alias("bad")
)


def normalize_score(pos_col, min_val, max_val):
    normalized_score = (col(pos_col) - col(min_val)) / (col(max_val) - col(min_val))
    normalized_score = (normalized_score * 10)
    # Handle null values
    normalized_score = when(normalized_score.isNull() | col(min_val).isNull() | col(max_val).isNull(), 
                            0).otherwise(normalized_score)
    return normalized_score

def clamp_score(score):
    return when(score < 1, 1).when(score > 5, 5).otherwise(score)

normalized_df = grouped_df.select(
    "lecture_id",
    (clamp_score(normalize_score("mean_diff_pos_score", "min_diff_pos_score", "max_diff_pos_score") -
     normalize_score("mean_diff_neg_score", "min_diff_neg_score", "max_diff_neg_score") + 3)).alias("level"),
    (clamp_score(normalize_score("mean_lec_pos_score", "min_lec_pos_score", "max_lec_pos_score") -
     normalize_score("mean_lec_neg_score", "min_lec_neg_score", "max_lec_neg_score") + 3)).alias("teaching_quality"),
    (clamp_score(normalize_score("mean_data_pos_score", "min_data_pos_score", "max_data_pos_score") -
     normalize_score("mean_data_neg_score", "min_data_neg_score", "max_data_neg_score") + 3)).alias("reference"),
    (clamp_score(normalize_score("mean_train_pos_score", "min_train_pos_score", "max_train_pos_score") -
     normalize_score("mean_train_neg_score", "min_train_neg_score", "max_train_neg_score") + 3)).alias("practice"),
    "good",
    "bad"
)

joined_df = normalized_df.join(df2, normalized_df["lecture_id"] == df2["course_identification"], "inner").drop("course_identification")
joined_df_with_inflearn = joined_df.withColumn("lecture_type", lit("INFLEARN")).withColumn("rating",joined_df["star_rate"].cast(DoubleType())).drop("star_rate")

# joined_df_with_inflearn.show(n=20, truncate=False)
# end_time = time.time()
# print("최종 정규화 시간:", end_time - start_time, "초")

# mysql_url = "jdbc:mysql://infou-db.c1egg6qaq55v.ap-northeast-2.rds.amazonaws.com:3306/infou_db"
# mysql_properties = {
#     "user": "root",
#     "password": "zxcasdqwe5",
#     "driver": "com.mysql.cj.jdbc.Driver"  # MySQL JDBC 드라이버
# }

# joined_df_with_inflearn.write.jdbc(url=mysql_url, table="lecture_detail", mode="overwrite", properties=mysql_properties)
# spark.stop()

joined_df_with_inflearn.write.format("csv").options(header='True', delimiter=',').mode('overwrite').save("spark_output/inflearn_course")