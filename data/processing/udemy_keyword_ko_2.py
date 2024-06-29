import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from konlpy.tag import Mecab

korean_development_keywords = {
    '파이썬', '자바', 'C', 'C++', 'C#', '자바스크립트', '타입스크립트', '루비', 'PHP', '스위프트', '코틀린',
    'HTML', 'CSS', 'Sass', 'Less', 'jQuery', 'AJAX', 'XML', 'JSON',
    '입문', '게임', '스크립트', '기획', 'RPG', 'Android', 'R',
    'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Oracle', 'SQLite', 'Redis', 'Cassandra',
    '데이터', '분석', '데이터분석', '데이터사이언스', '머신러닝', '딥러닝', '신경망',
    '인공지능', '자연어처리', '클라우드', 'AWS', 'Azure', 'Google', 'Cloud', 'GCP', 'IBM', 'Oracle',
    '데브옵스', '도커', '쿠버네티스', '젠킨스', '앤서블', '테라폼', '퍼펫', '셰프',
    'Git', 'GitHub', 'GitLab', 'Bitbucket', 'SVN',
    '애자일', '스크럼', '칸반', '린', 'XP', '익스트림 프로그래밍',
    '모바일', '안드로이드', 'iOS', '리액트 네이티브', '플러터', '자마린',
    '웹', '리액트', '앵귤러', '뷰', '스벨트', 'Next.js', 'Nuxt.js', 'Gatsby', '부트스트랩', '파운데이션', '마테리얼라이즈',
    '백엔드', '프론트엔드', '풀스택', 'API', 'REST', 'GraphQL', 'gRPC',
    'Node.js', '익스프레스', '장고', '플라스크', '스프링', '라라벨', '레일스', '신트라',
    '테스팅', '유닛 테스트', '통합 테스트', 'E2E 테스트', '셀레늄', 'Junit', 'Pytest', 'Mocha', 'Chai',
    '보안', '사이버 보안', '암호화', '해독', 'SSL', 'TLS', '방화벽',
    '네트워킹', 'TCP', 'UDP', 'IP', 'DNS', 'HTTP', 'HTTPS',
    '컨테이너', '가상화', 'Vagrant', 'VMware', 'VirtualBox',
    '디자인 패턴', '객체지향 프로그래밍', '함수형 프로그래밍', '반응형 프로그래밍',
    '빅데이터', '하둡', '스파크', '하이브', '피그', '카프카', '스톰', '플링크',
    '소프트웨어 엔지니어링', '시스템 설계', '아키텍처', '마이크로서비스', '모놀리틱', 'SOA',
    '사용자 경험', '사용자 인터페이스', 'UI', 'UX', '디자인', '프로토타이핑', '와이어프레임', 'Figma', '스케치', 'Adobe XD',
    '블록체인', '비트코인', '이더리움', '스마트 계약', '솔리디티', 'DApps',
    '게임 개발', '유니티', '언리얼 엔진', '게임 메이커', 'Godot',
    '로봇공학', '자동화', '사물인터넷', '라즈베리파이', '아두이노',
    '양자 컴양자 컴퓨팅', '양자역학', '양자알고리즘', '큐비트',
    '생물정보학', '생물공학', '유전체학', '단백체학', '시스템생물학',
    '비즈니스 인텔리전스', 'BI', 'Tableau', 'Power BI', 'QlikView', 'Looker',
    'CRM', 'Salesforce', 'HubSpot', 'Zoho', 'Microsoft Dynamics',
    'ERP', 'SAP', 'Oracle ERP', 'Microsoft Dynamics 365',
    'CMS', 'WordPress', 'Joomla', 'Drupal', 'Contentful', 'Strapi',
    '초보', '중급', '코딩', '머신', '러닝', '사이언스', '자연어', '오라클', '인공', '지능', 
    '시스템', '설계', '인터페이스', '사용자', '디자인', '객체', '프로그래밍', '알고리즘', '깃', '깃허브', '허브',
    'Machine', '기초', '구글', 'Boot'
    
    }
df = pd.read_csv("udemy_ko.csv")

df['keyword_array'] = df['keywords'].str.split(', ')
df['unique_keyword_array'] = df['keyword_array'].apply(lambda x: list(set(x)))

new_df = df[['id', 'unique_keyword_array']].copy()

mecab = Mecab()

def extract_stems(text):
    if text is None:
        return []
    
    stems = [word for word, pos in mecab.pos(text) if pos.startswith('N')]
    
    return stems

new_df['stems'] = new_df['unique_keyword_array'].apply(lambda x: extract_stems(' '.join(x)))


df['unique_keyword_array'] = df['stems']
df.drop(columns=['stems'], inplace=True)

def remove_duplicates(arr):
    return list(set(arr))

df['unique_keyword_array'] = df['unique_keyword_array'].apply(remove_duplicates)

df = new_df


vectorizer = TfidfVectorizer()

df['keyword_string'] = df['unique_keyword_array'].apply(lambda x: ', '.join(x))
tfidf_matrix = vectorizer.fit_transform(df['keyword_string'])

feature_names = vectorizer.get_feature_names_out()


tfidf_scores = defaultdict(lambda: defaultdict(float))
for i, row in enumerate(tfidf_matrix):
    for j, score in zip(row.indices, row.data):
        tfidf_scores[df['id'][i]][feature_names[j]] = score


def extract_top_keywords(keywords, dev_keywords, tfidf_score_dict, top_n=5):
    
    split_keywords = [kw.strip() for kw in keywords.split(',')]
    
    
    top_keywords = [kw for kw in split_keywords if kw in dev_keywords]
    
    
    if len(top_keywords) < top_n:
        remaining_keywords = [kw for kw in split_keywords if kw not in top_keywords]
        remaining_keywords = sorted(remaining_keywords, key=lambda x: tfidf_score_dict.get(x, 0), reverse=True)
        top_keywords.extend(remaining_keywords[:top_n - len(top_keywords)])
    
    return top_keywords[:top_n]


df['top_keywords'] = df.apply(lambda row: extract_top_keywords(row['keyword_string'], korean_development_keywords, tfidf_scores[row['id']]), axis=1)


new_df['topword1'] = ''
new_df['topword2'] = ''
new_df['topword3'] = ''
new_df['topword4'] = ''
new_df['topword5'] = ''

for index, row in new_df.iterrows():
    top_words = row['top_keywords']
    for i, top_word in enumerate(top_words, start=1):
        new_df.at[index, f'topword{i}'] = top_word



selected_df = new_df[['id', 'topword1', 'topword2', 'topword3', 'topword4', 'topword5']]
selected_df.to_csv('lecture_tag_ko.csv', index=False)
