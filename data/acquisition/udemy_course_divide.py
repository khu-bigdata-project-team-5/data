import pandas as pd
import os

# Load the DataFrame
df = pd.read_csv('Comments.csv')

# Define the chunk size
chunk_size = 500000

# Define the output directory for the CSV files
output_dir = './csv_grouped_chunks/'
os.makedirs(output_dir, exist_ok=True)

# Remove commas from the 'comment' column
df['comment'] = df['comment'].str.replace(',', '', regex=False)

# Group by 'course_id' without applying any aggregation function
grouped = df.groupby('course_id')

# Initialize variables
current_chunk = pd.DataFrame()  # Current chunk being accumulated
chunk_idx = 1  # Index for chunk file names

# Iterate over the groups
for course_id, group_df in grouped:
    # Concatenate the current group with the current chunk
    current_chunk = pd.concat([current_chunk, group_df])
    # Check if the current chunk size exceeds the threshold
    if len(current_chunk) >= chunk_size:
        # Save the current chunk to a CSV file
        current_chunk.to_csv(os.path.join(output_dir, f"chunk_{chunk_idx}.csv"), index=False)
        # Reset the current chunk and increment the chunk index
        current_chunk = pd.DataFrame()
        chunk_idx += 1

# Check if there are any remaining rows in the current chunk
if not current_chunk.empty:
    # Save the remaining rows as the last chunk
    current_chunk.to_csv(os.path.join(output_dir, f"chunk_{chunk_idx}.csv"), index=False)

print("Data has been grouped and saved in chunks successfully.")
