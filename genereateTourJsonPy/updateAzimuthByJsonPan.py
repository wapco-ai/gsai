import pandas as pd  
import json  

# Load CSV file  
csv_file_path = r'D:\AI\imageProccessingAI\pycode\genereateTourJsonPy\detected_objects_with_azimuth_latlon18.csv'  # Replace with your actual CSV file path  
data = pd.read_csv(csv_file_path)  

# Load JSON file  
json_file_path = r'D:\AI\imageProccessingAI\pycode\genereateTourJsonPy\nodesBB.json'  # Replace with your actual JSON file path  
with open(json_file_path, 'r') as f:  
    json_data = json.load(f)  

# Create a dictionary to map 'name' to 'pan'  
pan_dict = {item['name']: int(item['sphereCorrection']['pan'].replace('deg', '').strip()) for item in json_data}  

# Update the azimuth in the DataFrame  
def adjust_azimuth(row):  
    # Calculate the new azimuth value  
    new_azimuth = row['azimuth'] - pan_dict.get(row['file_name'], 0)  
    # If the new azimuth is negative, add 360 to wrap it  
    if new_azimuth < 0:  
        new_azimuth += 360  
    return new_azimuth  

data['azimuth'] = data.apply(adjust_azimuth, axis=1)  

# Save the updated DataFrame to a new CSV file  
updated_csv_file_path = r'D:\AI\imageProccessingAI\pycode\genereateTourJsonPy\detected_objects_with_azimuth_latlon18_updated.csv'  # Specify the path where you want to save the updated CSV  
data.to_csv(updated_csv_file_path, index=False)  

print(f"Updated CSV saved to {updated_csv_file_path}.")