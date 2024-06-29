import os
import requests
import pandas as pd
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 입력 CSV 파일 경로
input_path = "./dataset/Course_category_split.csv"
# 출력 CSV 파일 경로
output_path = "./dataset/Course_thumbnail.csv"

# Udemy API 기본 URL 및 헤더
base_url = "https://www.udemy.com/api-2.0/courses/"
headers = {
    "Authorization": "Basic YmlndG9nMDYwMkBnbWFpbC5jb206bUBzdGVyMDMwNQ=="
}

# 입력 CSV 파일을 읽어들임
input_df = pd.read_csv(input_path)
id_list = input_df['id'].tolist()

# 출력 CSV 파일이 이미 존재하는지 확인
file_exists = os.path.isfile(output_path)
if file_exists:
    check_df = pd.read_csv(output_path)
    exist_id_set = set(check_df['id'].tolist())
else:
    exist_id_set = set()

def fetch_course_data(course_id):
    if course_id in exist_id_set:
        print(f"Course ID {course_id} already exists in the output CSV file")
        return None
    url = f"{base_url}{int(course_id)}/"
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            thumbnail = data.get('image_480x270')
            if thumbnail:
                row = input_df[input_df['id'] == course_id].iloc[0].tolist()
                row.append(thumbnail)
                return row
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            print(f"Rate limit exceeded for course ID {course_id}, retrying after {retry_after} seconds")
            time.sleep(retry_after)
        elif response.status_code in [403, 404]:
            default_image_url = "https://www.udemy.com/staticx/udemy/images/v7/logo-udemy.svg"
            row = input_df[input_df['id'] == course_id].iloc[0].tolist()
            row.append(default_image_url)
            return row
        elif response.status_code == 503:
            print(f"Service unavailable for course ID {course_id}, retrying after 10 seconds")
            time.sleep(10)
        else:
            print(f"Failed to fetch data for course ID {course_id}: {response.status_code}")
            return None

# CSV 파일에 데이터를 실시간으로 추가할 수 있도록 설정
with open(output_path, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # 파일이 존재하지 않는 경우에만 헤더를 작성
    if not file_exists:
        writer.writerow(['id', 'title', 'is_paid', 'price', 'headline', 'num_subscribers', 'avg_rating', 
                         'num_reviews', 'num_comments', 'num_lectures', 'content_length_min', 'published_time', 
                         'last_update_date', 'category', 'subcategory', 'topic', 'language', 'course_url', 
                         'instructor_name', 'instructor_url', 'thumbnail'])

    # ThreadPoolExecutor를 사용하여 병렬로 API 요청을 보냄
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(fetch_course_data, course_id): course_id for course_id in id_list}
        
        for future in as_completed(future_to_id):
            result = future.result()
            if result:
                writer.writerow(result)
                print(f"Successfully fetched data for course ID {future_to_id[future]}")

print(f"CSV 파일이 성공적으로 {output_path}에 저장되었습니다.")
