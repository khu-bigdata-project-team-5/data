import requests
import pandas as pd

def fetch_and_save_data(api_key, endpoint, params):
    # API 요청
    response = requests.get(endpoint, params=params)
    data = response.json()

    # JSON 데이터에서 필요한 부분 추출하여 DataFrame 생성
    # 'items'는 실제 데이터가 포함된 부분의 키라고 가정
    items = data['response']['body']['items']
    df = pd.DataFrame(items)

    # 데이터 전처리 예시
    # 'date' 열이 있다고 가정하고 날짜 형식을 변환
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # 불필요한 데이터 추출
    if 'unnecessary_column' in df.columns:
        df.drop(columns=['unnecessary_column'], inplace=True)

    # DataFrame to CSV
    df.to_csv('output_data.csv', index=False)
    print("Data saved to 'output_data.csv'.")

# API Setting
api_key = 'your_api_key_here'
endpoint = 'https://api.publicdata.go.kr/your_api_endpoint'
params = {
    'serviceKey': api_key,
    'pageNo': '1',
    'numOfRows': '100',
    'type': 'json'
}


fetch_and_save_data(api_key, endpoint, params)

