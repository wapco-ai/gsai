import geopandas as gpd  
import pandas as pd  

# Load the shapefile  
shapefile_path = r'D:\AI\imageProccessingAI\shp\taged_images.shp'  # Update with the path to your shapefile  
gdf = gpd.read_file(shapefile_path)  

# Load the CSV file  
csv_file_path = r'D:\AI\imageProccessingAI\pycode\genereateTourJsonPy\detected_objects_with_azimuth18.csv'  # Update with your CSV file path  
df = pd.read_csv(csv_file_path)  

# Ensure the filename column is in the correct format in both DataFrames  
#gdf['filename'] = f"{gdf['filename']}.JPG"
gdf['filename'] = gdf['filename'].astype(str)  

#df['file_name'] = df['file_name'].astype(str)  
df['file_name'] = df['file_name'].str.replace('.JPG', '', case=False)  

# Merge the two DataFrames on the filename  
merged_df = pd.merge(df, gdf[['filename', 'longitude', 'latitude']],   
                     left_on='file_name',   
                     right_on='filename',   
                     how='left')  

# Drop the redundant 'filename' column after merge  
merged_df.drop(columns=['filename'], inplace=True)  

# Save the result to a new CSV file  
output_csv_path = r'D:\AI\imageProccessingAI\pycode\genereateTourJsonPy\detected_objects_with_azimuth_latlon18.csv'  
merged_df.to_csv(output_csv_path, index=False)  

print("Successfully merged longitude and latitude. Output saved to detected_objects_with_azimuth_latlon18.csv.")