#Get Inflearn course data using the Inflearn API
import pandas as pd
import requests
import json

# Base URL for the API
base_url = "https://www.inflearn.com/courses/client/api/v1/course/search"

# Parameters for the API
params = {
    "isDiscounted": "false",
    "isNew": "false",
    "pageSize": 30,
    "types": "ONLINE"
}

# List to store course data
courses_data = []

# Iterate through page numbers from 1 to 113
for page_number in range(1, 115):
    # Update the page number in the parameters
    params["pageNumber"] = page_number
    
    # Make the request to the API
    response = requests.get(base_url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Extract data from the response
        items = response.json().get('data', {}).get('items', [])
        
        # Extract course and instructor data
        for item in items:
            course_info = item['course']
            instructor_info = item['instructor']
            
            # Mapping API fields to database schema fields
            course_data = {
                'lecture_id': course_info['id'],
                'slug': course_info['slug'],
                'thumbnailUrl': course_info['thumbnailUrl'],
                'title': course_info['title'],
                'description': course_info['description'],
                'reviewCount': course_info['reviewCount'],
                'studentCount': course_info['studentCount'],
                'likeCount': course_info['likeCount'],
                'star': course_info['star'],
                'isExclusive': course_info['isExclusive'],
                'isNew': course_info['isNew'],
                'isUpdated': course_info['isUpdated'],
                'updatedAt': course_info['updatedAt'],
                'publishedAt': course_info['publishedAt'],
                'instructor_id': instructor_info['id'],
                'instructor_name': instructor_info['name'],
                # Initialize metadata columns
                'level': None,
                'skills': None,
                'parentCategories': None,
                'childCategories': None,
                'price':None
            }
            
            # Process metadata
            metadata = course_info.get('metadata', {})
            course_data['level'] = metadata.get('level')
            course_data['skills'] = ', '.join(metadata.get('skills', []))
            course_data['parentCategories'] = ', '.join(metadata.get('parentCategories', []))
            course_data['childCategories'] = ', '.join(metadata.get('childCategories', []))

            list_price = item['listPrice']
            course_data['price'] = list_price.get('payPrice')
            
            courses_data.append(course_data)

# Create a DataFrame from the collected data
df = pd.DataFrame(courses_data)

# Save the DataFrame to a CSV file
csv_file_path = 'inflearn_course_data.csv'
df.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")
