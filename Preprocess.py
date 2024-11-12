import pandas as pd
import os
import glob
input_folder = "C:/Users/Lenovo/OneDrive/Desktop/Main Project/Feb Data"
output_folder = "C:/Users/Lenovo/OneDrive/Desktop/Processed Data"

os.makedirs(output_folder, exist_ok=True)
excel_files = glob.glob(os.path.join(input_folder, "0[1-9].02.2024_AFC-KMRL-Data_-_*.xlsx")) + \
              glob.glob(os.path.join(input_folder, "1[0-9].02.2024_AFC-KMRL-Data_-_*.xlsx")) + \
              glob.glob(os.path.join(input_folder, "2[0-9].02.2024_AFC-KMRL-Data_-_*.xlsx"))

for file_path in excel_files:
    df = pd.read_excel(file_path)
    df['Transaction Time'] = pd.to_datetime(df['Transaction Time']).dt.time
    df['Date'] = pd.to_datetime(df['Date']).dt.date.astype(str)
    df['Fare Product'] = df['Fare Product'].str.strip().str.lower()
    excluded_products = ["period pass", "staff card", "station access card"]
    filtered_data = df[~df['Fare Product'].isin(excluded_products)]
    filtered_data = filtered_data.dropna()
    New = filtered_data[filtered_data['Transaction Type'].isin(["Entry", "Exit"])]
    columns_to_drop = ["Equipment Type", "Equipment ID", "Fare Media", "Ticket/Card Number", "Fare"]
    New = New.drop(columns=columns_to_drop, errors='ignore')
    
   
    output_file_path = os.path.join(output_folder, os.path.basename(file_path).replace(".xlsx", "_processed.xlsx"))
    
  
    New.to_excel(output_file_path, index=False)
    

    print(f"Processed and saved: {output_file_path}")


