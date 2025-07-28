import pandas as pd 
import os  

script_path = os.path.abspath(__file__)  # Get the absolute path of the script  
script_dir = os.path.split(script_path)[0]  # Get the directory of the script
csvout_filename = 'detected_objects_with_azimuth18.csv'  
csvout_file_path = os.path.join(script_dir, csvout_filename) 
 
csv_filename = 'detected_texts.csv'  
csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_outMultiTask')) 
csv_file_path = os.path.join(script_dir, csv_filename)

def calculate_azimuth(obj_x, image_width):  
    # Calculate azimuth based on the x coordinate  
    azimuth = (obj_x - (image_width / 2)) / image_width * 360  
    # Normalize the azimuth to the range [0, 360]  
    if azimuth < 0:  
        azimuth += 360  
    return azimuth  

# Load your CSV file  
#csv_file_path = r'D:\AI\imageProccessing\pano_app_test\py\genereate_tour_json\result.csv'  # Update with your CSV file path  
df = pd.read_csv(csv_file_path)  

# Initialize a list to hold results  
results = []  

# Assume predefined image dimensions (you can replace these with actual values if known)  
image_width = 5760  # Example width  
image_height = 2880  # Example height  

# Iterate through each row in the DataFrame  
for index, row in df.iterrows():  
    # Read center coordinates and parse them  
    center_coordinates = row['center_coordinates']  
    # Extract x and y from "(x, y)" format  
    #obj_x, obj_y = map(int, center_coordinates.strip("()").split(", "))  
    obj_x, obj_y = map(int, center_coordinates.strip("()").split(","))  
    
    # Calculate the azimuth angle  
    azimuth_angle = calculate_azimuth(obj_x, image_width)  
    
    # Append filename and azimuth angle to results  
    results.append({  
        "file_name": row['file_name'],  
        "bbox": row['bbox'],  
        "center_coordinates": row['center_coordinates'],  
        "detected_text": row['detected_text'],  
        "azimuth": azimuth_angle  
    })  

# Create a new DataFrame for results  
results_df = pd.DataFrame(results)  

# Optionally, save results to a new CSV file  
results_df.to_csv(csvout_file_path, index=False)  

print("Azimuth calculation completed. Results saved to detected_objects_with_azimuth18.csv.")