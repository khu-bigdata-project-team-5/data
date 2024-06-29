import pandas as pd

# 데이터 불러오기
user_lecture_df = pd.read_csv('dataset/user_lecture.csv')
udemy_user_df = pd.read_csv('dataset/udemy_user.csv')
lecture_tag_df = pd.read_csv('dataset/lecture_tag_combined.csv')

# 'inflearn' 플랫폼 스킵, 'udemy' 플랫폼만 남기기
user_lecture_df = user_lecture_df[user_lecture_df['platform'] == 'udemy']

# inner join 수행
user_tag_df = pd.merge(user_lecture_df, lecture_tag_df, left_on='lecture_id', right_on='id', how='inner')

# 태그 가중치 설정
weights = {'topword1': 3, 'topword2': 2.5, 'topword3': 2, 'topword4': 1.5, 'topword5': 1}

# 태그 가중치 적용
for topword, weight in weights.items():
    user_tag_df[topword] = user_tag_df[topword].apply(lambda x: (x, weight))

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

# 유저별 상위 5개의 태그 추출
user_topwords = []
for (user_id, platform, name), tags in user_tag_weighted.items():
    sorted_tags = sorted(tags.items(), key=lambda item: item[1], reverse=True)[:5]
    topwords = [tag for tag, _ in sorted_tags]
    topwords = [tag for tag in topwords if tag is not None]  # None 값 제거
    topwords += [None] * (5 - len(topwords))  # 길이를 5로 맞추기 위해 None 추가
    user_topwords.append([user_id, platform, name] + topwords)

# 결과 데이터프레임 생성
columns = ['user_id', 'platform', 'name', 'topword1', 'topword2', 'topword3', 'topword4', 'topword5']
result_df = pd.DataFrame(user_topwords, columns=columns)

# 최종 필터링으로 중복된값있으면 하나는 제거
result_df = result_df.drop_duplicates(subset='user_id', keep='first')

result_df.to_csv('dataset/user_keyword.csv', index=False, encoding='utf-8-sig')