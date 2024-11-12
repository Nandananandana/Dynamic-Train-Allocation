import pandas as pd
import glob
import math

folder_path = "C:/Users/Lenovo/OneDrive/Desktop/Processed Data"
file_paths = glob.glob(f'{folder_path}/*.xlsx')
data_frames = []


for file in file_paths:
    df = pd.read_excel(file)
    data_frames.append(df)
    print(f"File {file} has been read and added to the combined DataFrame.")


combined_df = pd.concat(data_frames, ignore_index=True)

# Define chunk size to stay within Excel's row limit
chunk_size = 1_000_000

output_template = "C:/Users/Lenovo/OneDrive/Desktop/combined_february_part_{}.xlsx"

# Split combined_df into chunks and save each chunk as a new file
for i in range(0, len(combined_df), chunk_size):
    chunk = combined_df[i:i + chunk_size]
    output_path = output_template.format(i // chunk_size + 1)
    chunk.to_excel(output_path, index=False)
    
    # Print statement to confirm each file creation
    print(f"File created and saved: {output_path}")

print("All files have been processed and saved.")

