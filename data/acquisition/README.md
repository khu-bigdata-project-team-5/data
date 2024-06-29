# INFoU Data Acquisition

This is the directory for all the crawling of Udemy & Inflearn Lectures & Comments

## Udemy
1. udemy_crawling.ipynb (연습 코드)
	- 유데미 댓글 데이터 크기가 너무 크기 때문에 이를 분리 및 크롤링 등의 연습을 위한 코드
2. udemy_course_with_thumnail.py (실제 코드)
	- Udemy API를 이용하 기존 kaggle 에는 존재하지 않던 course의 썸네일 이미지를 추가하기 위한 과정
	- 예외 처리를 통한 중간의 에러가 존재하더라도 종료 없이 재시도
	- threadpool을 통해 10배 빠른 데이터 처리 속도
	``` python
     # ThreadPoolExecutor를 사용하여 병렬로 API 요청을 보냄
     with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(fetch_course_data, course_id): course_id for course_id in id_list}
        
        for future in as_completed(future_to_id):
            result = future.result()
            if result:
                writer.writerow(result)
                print(f"Successfully fetched data for course ID {future_to_id[future]}")

     ```
	
3. udemy_comment_divide.py (실제 코드)
	 - (1)에서 정제된 코드로 시도해 본 결과, 댓글이 강의별로 묶여 있지 않은 문제를 해결하고자 만든 실제 사용 코드
	 - (1)의 코드에서 groupBy를 통해 course_id 별로 묶어, 최종 댓글 코드 생성
	 
     
     
## Inflearn
1. inflearn_course_crawling_selenium.ipynb
	- 인프런 강의 ui 가 바뀌기 전에 사용하였던 selenium을 이용한 Inflearn 강좌를 가지고 오는 코드
2. inflearn_lecture.py (실제 코드)
	- 인프런 강의 ui가 바뀌고 나서 개선한 코드. 
	- 실제 인프런 홈페이지에서 API를 가지고 와서 데이터 추출하기 때문에, 기존 selenium 코드보다 훨씬 빠른 속도로 진행 가능
3. inflearn_comment.py
	- 인프런 댓글 api에서 모든 댓글들을 들고오는 코드.
	- (2)에서 추출한 강의 id를 바탕으로 모든 댓글들을 가져옴
