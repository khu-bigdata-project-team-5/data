# Data Processing

- This is all about Data Processing of INFoU Service

목차

- [Lecture Analysis](#1-.Lectrue-Analysis)
- [Lecture Keyword](#2.-Lecture-Keyword)
- [Lectue Keyword CloudWord](#3.-Lecture-CloudWord)
- [User Keyword](#4.-User-Keyword)

## 1. Lecture Analysis

인프런/유데미의 댓글/강좌 데이터를 바탕으로 각 강좌 데이터를 분석합니다. (클러스터에서 진행)

### 분석 내용
   1. 강의 관련 분석
   - **강의력** : 강의를 얼마나 잘하는지에 대한 지표
   - **난이도** : 해당 강의가 얼마나 어려운지에 대한 지표
   - **강의자료** : 해당 강의의 강의자료가 잘 만들었는지에 대한 지표
   - **실습** : 실습 / 예제 등의 코드가 좋은지에 대한 지표
   - **평점** : 해당 강의에 대한 실제 평점
---
  2. 댓글 긍정/부정 분석
  		1. ~~BERT의 긍정 / 부정 모델 사용~~  
 
		너무 많은 시간이 걸림  및 우분투 서버에 안 올라가는 이유로 사용 배제
		
        2. 직접 구현
		
        앞선 강의 관련 분석에서 긍정적인 부분과 부정적인 부분의 값을 비교하여 긍정/부정 분석 진행
### 실제 코드

- inflearn_comment_processing / udemy_comment_processing.py

1.  파일 읽어오기 

	Spark Session 생성 및 hdfs 에서 파일 읽어오기.
    
    option 설명
	- ``.option("encoding", "UTF-8") \ ``  # 한글을 읽기 위한 encoding 옵션
	- ``.option("multiline", "true") \ `` # 댓글의 크기가 크기 때문에 필요한 옵션
	- ``.load("combined_course_reviews_2.csv") \ `` # HDFS 세팅이 되어있기에, 파일명 입력 시 불러오기 가능 
	

``` python 
spark = SparkSession.builder.appName("INFoU").getOrCreate()

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

```


2. 언어 감지
``` python
#영어 감지
def detect_language_langid(text):
    try:
        lang, _ = identifier.classify(text)
        return lang
    except:
        return "Unknown"

detect_language_udf = udf(detect_language_langid, StringType())
merged_course_info = df_filtered.withColumn("language", detect_language_udf(col("comment")))

filtered_course_info = merged_course_info.filter((col("language") == "en"))
```

3. 형태소 분석

	댓글들을 단어들의 집합으로 전처리 수행
    
    코드 설명
	- ``mecab = Mecab() ``  # 한글어 형태소 분석기 중 Mecab 사용
	- ``pos_tags = ['NNG', 'MAG', 'VV', 'SL', 'VA'] `` 형태소 중 필요한 형태소만 취득
    - ``tokenize_and_filter_udf = udf(tokenize_and_filter, ArrayType(StringType()))`` udf에 형태소 분석 함수 등록
``` python
from konlpy.tag import Mecab

mecab = Mecab()

pos_tags = ['NNG', 'MAG', 'VV', 'SL', 'VA']
def tokenize_and_filter(text):
    if text is None:
        return []
    tokens = [word for word, pos in mecab.pos(text) if pos in pos_tags]
    return tokens

tokenize_and_filter_udf = udf(tokenize_and_filter, ArrayType(StringType()))

tokenized_df = df.withColumn("words",tokenize_and_filter_udf(col("comment")))
```

4. **Word2Vec** 모델 사용
	
    Word2Vec 모델을 통해 단어들 벡터화
    
    ``` python
    word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="words", outputCol="result")
	model = word2Vec.fit(tokenized_df)
    ```

5. 분석할 내용 관련 단어 미리 정의 

	- 한국어 분석 키워드 

	``` python
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
    ```
    
    - 영어 분석 키워드
    ``` python
    difficulty_pos_keywords = ["beginner", "easy", "simple", "understand", "well", "good", "nice","recommend", "basic","Excellent"]
    difficulty_neg_keywords = ["difficult", "hard", "difficulty", "not", "lack","dont","bad","poor","worst","waste"]
    lecture_pos_keywords = ["understand", "recommend", "material", "content", "fundamental", "detailed", "presentation"]
    lecture_neg_keywords = ["careless", "lack", "difficulty"]
    data_pos_keywords = ["analogy", "introduction", "specification", "image", "animation", "document", "code", "content", "method", "process", "way", "search"]
    data_neg_keywords = ["inadequate", "lack", "difficulty"]
    train_pos_keywords = ["code", "example", "method", "image", "project", "development"]
    train_neg_keywords = ["inadequate", "lack", "difficulty"]
    ```
 6. **Cosine Similarity**를 이용한 키워드 분석
 	
    해당 키워드들을 word2Vec을 피팅한 모델에 넣어 cosine similarity 분석
    
    - 코드 분석 
    - ``synonyms = model.findSynonyms(word, len(word)).collect()`` 를 통해 cosine similarity 구행
    - ``similarity_score = sum(syn[1] for syn in synonyms if syn[1] > 0)``를 통해 cosine similarity 값들의 합을 추가해줌 

	- 키워드 별로 전부 수행 

 	``` python
    def compute_similarity(word):
    try:
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
    ```
 7. 분석한 값 연산 및 댓글 긍/부정 검사
 	
    (5)의 코드를 전 모델에 수행하며 실제 값 및 댓글 긍/부정의 최종 값들을 계산함
    
    코드 설명
    
    - ``def match_keywords(...) `` 각 댓글마다 수행하는 작업. 여기서 각 댓글에서 분석한 값이 나옴
    - `` pos_score = diff_pos_score + lec_pos_score + data_pos_score + train_pos_score `` 댓글 긍정 분석
    - ``analysis = 1.0 if pos_score >= neg_score else -1.0`` 최종 긍/부정 분류
    
    ``` python 
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
    ```
    
8. 정규화 진행

	값의 범위가 일정하지 않아 0~5 점사이의 값을 만들기 위하여 정규화 진행
    
    ``` python
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
    ```
    
9. 강좌 평점과 join 진행
```
joined_df = normalized_df.join(df2, normalized_df["lecture_id"] == df2["course_identification"], "inner").drop("course_identification")
joined_df_with_inflearn = joined_df.withColumn("lecture_type", lit("INFLEARN")).withColumn("rating",joined_df["star_rate"].cast(DoubleType())).drop("star_rate")
```

10. 최종 csv 파일로 저장
```
joined_df_with_inflearn.write.format("csv").options(header='True', delimiter=',').mode('overwrite').save("spark_output/inflearn_course")
```


## 2. Lecture Keyword
인프런/유데미의 강좌 데이터를 바탕으로 각 강좌의 키워드를 뽑아냅니다. (로컬 및 클러스터에서 진행)

### 분석 내용
   1. 강의 관련 분석
  
  		강의의 제목 및 헤드라인을 분석하여 해당하는 키워드 뽑아냄


### 실제 코드
- udemy_keyword_*.py

1. TF-IDF 모델에 대입

	Tfidf Vector 모델에 keywords 넣음
``` python
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['keywords_string'])
feature_names = vectorizer.get_feature_names_out()
```
2. 미리 정의한 키워드 dict 에 대입

	미리 정의한 키워드에 있는 키워드를 먼저 집어넣음
``` python
tfidf_scores = defaultdict(lambda: defaultdict(float))

for i, row in enumerate(tfidf_matrix):
    for j, score in zip(row.indices, row.data):
        tfidf_scores[df['id'][i]][feature_names[j]] = score
```
3. 랭킹에 따라 키워드 정리

	분류 기준에 따라 랭킹화함
``` python
def extract_top_keywords(keywords, dev_keywords, tfidf_score_dict, top_n=5):

    top_keywords = [kw for kw in keywords if kw in dev_keywords]
    

    if len(top_keywords) < top_n:
        remaining_keywords = [kw for kw in keywords if kw not in top_keywords]
        remaining_keywords = sorted(remaining_keywords, key=lambda x: tfidf_score_dict.get(x, 0), reverse=True)
        top_keywords.extend(remaining_keywords[:top_n - len(top_keywords)])
    
    return top_keywords[:top_n]
```



## 3. Lecture CloudWord

프로젝트 진행에 앞서 진행한 키워드 별 카테고리 클라우드워드 (로컬에서 진행)

코드 설명

1. 코드 정렬
``` python
# 텍스트를 정리하는 함수, 한글, 영어, 숫자, 공통 구두점 문자를 제외한 문자 제거
def clean_text(text):
    return re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9,.\s]', '', text)
```
2. 워크 클라우드 생성
``` python
# 한글 글꼴 경로 지정, 워드 클라우드 생성
font_path = r'C:\Users\高奥成\Desktop\comment\Nanum_Gothic\NanumGothic-Regular.ttf'  # 로컬 한글 글꼴 경로로 업데이트
wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(skills_text)

```

## 3. User Keyword

(2)에서 뽑은 Lecture Keyword를 바탕으로 사용자와 조인하여 커리큘럼 생성 (로컬에서 진행)

코드 설명

1. inner join 수행
``` python
# inner join 수행
user_tag_df = pd.merge(user_lecture_df, lecture_tag_df, left_on='lecture_id', right_on='id', how='inner')
```
2. 태그 가중치 계산
``` python
# 유저별 태그 가중치 합계 계산,
user_tag_weighted = {}
for index, row in user_tag_df.iterrows():
    user_id = row['user_id']
    platform = row['platform']
    name = row['name']
    if (user_id, platform, name) not in user_tag_weighted:
        user_tag_weighted[(user_id, platform, name)] = {}
    for topword in weights.keys():
        tag, weight = row[topword]
        if pd.notna(tag):
            if tag in user_tag_weighted[(user_id, platform, name)]:
                user_tag_weighted[(user_id, platform, name)][tag] += weight
            else:
                user_tag_weighted[(user_id, platform, name)][tag] = weight
```

3. 태그 추출
``` python
# 유저별 상위 5개의 태그 추출
user_topwords = []
for (user_id, platform, name), tags in user_tag_weighted.items():
    sorted_tags = sorted(tags.items(), key=lambda item: item[1], reverse=True)[:5]
    topwords = [tag for tag, _ in sorted_tags]
    topwords = [tag for tag in topwords if tag is not None]  # None 값 제거
    topwords += [None] * (5 - len(topwords))  # 길이를 5로 맞추기 위해 None 추가
    user_topwords.append([user_id, platform, name] + topwords)
```