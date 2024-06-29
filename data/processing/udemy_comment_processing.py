# Udemy 중 한글과 관련된 것만 수행


import math
from langid.langid import LanguageIdentifier, model
from pyspark.sql.functions import lit, udf, col, when, min, max, mean, count
from pyspark.sql.types import FloatType, StringType, ArrayType, DoubleType
from pyspark.ml.feature import Word2Vec, StopWordsRemover
import os
import time
import numpy as np
from pyspark.sql import SparkSession
from konlpy.tag import Mecab
mecab = Mecab()


os.environ["SPARK_HOME"] = "./spark-2.4.8-bin-hadoop2.7"
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
spark = SparkSession.builder.appName("INFoU").getOrCreate()

start_time = time.time()

df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .option("multiLine", "true") \
            .option("quote", '"') \
            .load("merged_output.csv")
            # .load(f"csv_grouped_chunks/chunk_{i}.csv")


for col_name in df.columns:
    df = df.withColumnRenamed(col_name, col_name.strip())

df = df.select("course_id", "comment")

df2 = spark.read.format("csv") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .option("encoding", "UTF-8") \
                .option("multiline", "true") \
                .option("quote", '"') \
                .load("Course_info_with_thumbnail.csv").select("id", "avg_rating")

# Filter df to only include rows with course_id present in df2
df_filtered = df.join(df2, df["course_id"] == df2["id"], "inner").drop("id")
df_filtered = df_filtered.withColumn("rating", df2["avg_rating"].cast(DoubleType()))
df_filtered = df_filtered.filter(df_filtered["course_id"].isNotNull())

def detect_language_langid(text):
    try:
        lang, _ = identifier.classify(text)
        return lang
    except:
        return "Unknown"

detect_language_udf = udf(detect_language_langid, StringType())
merged_course_info = df_filtered.withColumn("language", detect_language_udf(col("comment")))

filtered_course_info = merged_course_info.filter((col("language") == "en"))

def tokenize_and_filter_en(text: str):
    if text is None:
        return []
    tokens = text.split(" ")  # 영어 텍스트를 띄어쓰기 단위로 분리
    return tokens

pos_tags = ['NNG', 'MAG', 'VV', 'SL', 'VA']
def tokenize_and_filter_ko(text):
    if text is None:
        return []
    tokens = [word for word, pos in mecab.pos(text) if pos in pos_tags]
    return tokens

tokenize_and_filter_en_udf = udf(tokenize_and_filter_en, ArrayType(StringType()))
tokenized_df_en = df_filtered.withColumn("words", tokenize_and_filter_en_udf(col("comment"))).drop("comment")

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(tokenized_df_en).drop("words")

word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="filtered_words", outputCol="result")
model = word2Vec.fit(cleaned_df)

word_vectors = {row['word']: row['vector'].toArray() for row in model.getVectors().collect()}

def compute_similarity(word):
    try:
        vec = word_vectors.get(word, np.zeros(10))
        synonyms = model.findSynonyms(vec, len(word)).collect()
        similarity_score = sum(syn[1] for syn in synonyms if syn[1] > 0)
        return similarity_score
    except:
        return 0.0

def precompute_keyword_similarities(keywords):
    return {keyword: compute_similarity(keyword) for keyword in keywords}

# difficulty_pos_keywords = ["beginner", "easy", "simple", "understand", "well", "good", "nice","recommend", "basic","Excellent"]
# difficulty_neg_keywords = ["difficult", "hard", "difficulty", "not", "lack","dont","bad","poor","worst","waste"]
# lecture_pos_keywords = ["understand", "recommend", "material", "content", "fundamental", "detailed", "presentation"]
# lecture_neg_keywords = ["careless", "lack", "difficulty"]
# data_pos_keywords = ["analogy", "introduction", "specification", "image", "animation", "document", "code", "content", "method", "process", "way", "search"]
# data_neg_keywords = ["inadequate", "lack", "difficulty"]
# train_pos_keywords = ["code", "example", "method", "image", "project", "development"]
# train_neg_keywords = ["inadequate", "lack", "difficulty"]
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

match_keywords_udf = udf(lambda words: match_keywords(words, difficulty_pos_similarities, difficulty_neg_similarities, lecture_pos_similarities, lecture_neg_similarities, data_pos_similarities, data_neg_similarities, train_pos_similarities, train_neg_similarities), ArrayType(FloatType()))

result_df = cleaned_df.withColumn("scores", match_keywords_udf(col("filtered_words"))) \
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
                        .drop("language") \
                        .drop("filtered_words")

cnt_condition = lambda cond: sum(when(cond, 1).otherwise(0))
grouped_df = result_df.groupBy("course_id").agg(
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
    count(when(col("analysis") == -1.0, True)).alias("bad"),
)
grouped_df = grouped_df.join(result_df.select("course_id", "rating").distinct(), "course_id", "left")

def normalize_score(pos_col, min_val, max_val):
    normalized_score = (col(pos_col) - col(min_val)) / (col(max_val) - col(min_val))
    normalized_score = (normalized_score * 10)
    normalized_score = when(normalized_score.isNull() | col(min_val).isNull() | col(max_val).isNull(), 
                            0).otherwise(normalized_score)
    return normalized_score

def clamp_score(score):
    return when(score < 1, 1).when(score > 5, 5).otherwise(score)

normalized_df = grouped_df.select(
    col("course_id").alias("lecture_udemy_id"),
    (clamp_score(normalize_score("mean_diff_pos_score", "min_diff_pos_score", "max_diff_pos_score") -
    normalize_score("mean_diff_neg_score", "min_diff_neg_score", "max_diff_neg_score") + 3)).alias("level"),
    (clamp_score(normalize_score("mean_lec_pos_score", "min_lec_pos_score", "max_lec_pos_score") -
    normalize_score("mean_lec_neg_score", "min_lec_neg_score", "max_lec_neg_score") + 3)).alias("teaching_quality"),
    (clamp_score(normalize_score("mean_data_pos_score", "min_data_pos_score", "max_data_pos_score") -
    normalize_score("mean_data_neg_score", "min_data_neg_score", "max_data_neg_score") + 3)).alias("reference"),
    (clamp_score(normalize_score("mean_train_pos_score", "min_train_pos_score", "max_train_pos_score") -
    normalize_score("mean_train_neg_score", "min_train_neg_score", "max_train_neg_score") + 3)).alias("practice"),
    "rating",
    "good",
    "bad"
)
normalized_df.write.format("csv").options(header='True', delimiter=',').mode('overwrite').save(f"spark_output/udemy_course_ko")
