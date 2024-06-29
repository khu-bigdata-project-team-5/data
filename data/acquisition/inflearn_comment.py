import requests
import pandas as pd
import time

# Function to fetch data from a single page for a given link and id
def fetch_page_data(link, page_number, page_size):
    try:
        url = f"https://www.inflearn.com/api/v2/review/course/{link}?pageSize={page_size}&pageNumber={page_number}&isNew=true"
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

# Function to process the data and create a DataFrame
def create_dataframe(links_with_ids, page_size):
    user_id = []
    username = []
    rating = []
    date = []
    comment = []
    is_reply = []
    course_id = []

    # Iterate through each link and its id
    # for item in links_with_ids[665:]:
        # link = item["link"]
        # course_id_value = item["id"]
    

    initial_data = fetch_page_data(links_with_ids, 1, page_size)
    # if not initial_data:
    #     continue 
    total_pages = initial_data['data']['totalPage']

    # Iterate through each page to collect data
    for page_number in range(1, total_pages + 1):
        data = fetch_page_data(links_with_ids, page_number, page_size)
        items = data['data']['items']

        for review in items:
            user_id.append(review['userId'])
            username.append(review['userName'])
            rating.append(review['star'])
            date.append(review['createdAt'])
            comment.append(review['body'])
            is_reply.append(True if review['comments'] else False)
            course_id.append(links_with_ids)

    # Create the DataFrame
    df = pd.DataFrame({
        'user_id': user_id,
        'username': username,
        'rating': rating,
        'date': date,
        'comment': comment,
        'isReply': is_reply,
        'lecture_id': course_id
    })

    return df

# Main function to create the DataFrame and save it to a CSV file
def main():
    page_size = 167

    df = pd.read_csv('inflearn_course_data.csv')
    links = df['lecture_id'].tolist()
    
    # Create DataFrame
    df = create_dataframe(links, page_size)
    
    # Save DataFrame to CSV
    df.to_csv('reviews_data1.csv', index=False)
    
    return df

# Run the main function and display the DataFrame
df = main()
df.head()
