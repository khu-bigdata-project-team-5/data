import pandas as pd
import os

# CSV 파일이 있는 디렉토리 지정
csv_dir_path = "/Users/yh/Desktop/bigdata-class/class/mid/datas_csv"

# 모든 CSV 파일을 담을 빈 데이터프레임 생성
combined_csv = pd.DataFrame()

# 디렉토리 내의 모든 CSV 파일을 반복 처리
for file_name in os.listdir(csv_dir_path):
    if file_name.endswith('.csv'):
        # CSV 파일 경로 구성
        file_path = os.path.join(csv_dir_path, file_name)
        # CSV 파일 읽기
        data_frame = pd.read_csv(file_path, encoding='cp949')
        # 데이터프레임 병합
        combined_csv = pd.concat([combined_csv, data_frame], ignore_index=True)

# 결과 데이터프레임 확인
# print(combined_csv.head())


# 필요하다면, 병합된 데이터를 새로운 CSV 파일로 저장
combined_csv.to_csv((csv_dir_path +'/combined_csv.csv'), index=False)
