import geopandas as gpd  
import pandas as pd  
from shapely.geometry import LineString  
import math  

# Load the shapefile containing parcels  
parcel_shapefile_path = r'D:\AI\imageProccessingAI\shp\shp\ghataat.shp'  
parcels_gdf = gpd.read_file(parcel_shapefile_path)  

# Change to an appropriate UTM CRS for distance calculations  
parcels_gdf = parcels_gdf.to_crs(epsg=32640)  # EPSG code for UTM   

# Load the CSV file where positions are stored   
csv_file_path = r'D:\AI\imageProccessingAI\pycode\genereateTourJsonPy\detected_objects_with_azimuth_latlon18_updated.csv'  
df = pd.read_csv(csv_file_path)  
#df = df[df['confidence'] > 0.1]  

def create_line(longitude, latitude, azimuth, length=0.0002):  
    """Create a LineString from a point given azimuth.""" 
    #radians = math.radians(azim)  
    end_x = longitude + (length * math.cos(azimuth))  
    end_y = latitude + (length * math.sin(azimuth))  
    return LineString([(longitude, latitude), (end_x, end_y)])  

# Initialize results  
results = []  
selected_parcels = []  
lines = []  # To store created lines  
max_nearest_count = 1  # Specify how many nearest parcels you want to retrieve  

# Iterate through each row in the DataFrame  
for index, row in df.iterrows():  
    longitude = row['longitude']  
    latitude = row['latitude']  
    azimuth = row['azimuth']  

    line = create_line(longitude, latitude, azimuth)  
    lines.append(line)  # Append the created line to the list  

    # Convert line to a GeoSeries and re-project it to the same CRS as parcels_gdf  
    line_gdf = gpd.GeoSeries([line], crs="EPSG:4326").to_crs(epsg=32640)  # Direct conversion to UTM  

    # Find parcels that intersect the line  
    nearby_parcels = parcels_gdf[parcels_gdf.intersects(line_gdf.geometry[0])].copy()  # Create a new copy  

    if not nearby_parcels.empty:  
        # Calculate distances from the line to each parcel  
        nearby_parcels['distance'] = nearby_parcels.geometry.distance(line_gdf.geometry[0])  
        
        # Sort parcels by distance and select the nearest ones  
        nearest_parcels = nearby_parcels.nsmallest(max_nearest_count, 'distance')  

        parcel_name = nearest_parcels['Name'].tolist()  # Extract the names of the nearest parcels  
        selected_parcels.extend(nearest_parcels.geometry)  # Store the geometries of the nearest parcels  
    else:  
        parcel_name = None  

    # Append results to the list  
    results.append({  
        "file_name": row['file_name'],  
        "detected_text": row['detected_text'],  
        "bbox": row['bbox'],  
        "center_coordinates": row['center_coordinates'],  
        "azimuth": azimuth,  
        "longitude": longitude,  
        "latitude": latitude,  
        "parcel_name": parcel_name  
    })  

# Create a new DataFrame for results  
results_df = pd.DataFrame(results)  

# Save the result to a new CSV file  
output_csv_path = r'D:\AI\imageProccessingAI\pycode\genereateTourJsonPy\detected_objects_with_parcelnames18.csv'  
results_df.to_csv(output_csv_path, index=False)  

# Create a GeoDataFrame from the selected parcels  
if selected_parcels:  
    selected_parcels_gdf = gpd.GeoDataFrame(geometry=selected_parcels)   
    # Set the CRS directly to EPSG:32640  
    selected_parcels_gdf.set_crs(epsg=32640, inplace=True)  

    # Save the selected parcels to a new shapefile  
    output_shapefile_path = r'D:\AI\imageProccessingAI\shp\shp\ghataat_selected18.shp'  
    selected_parcels_gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')  

# Create a GeoDataFrame for the lines  
lines_gdf = gpd.GeoDataFrame(geometry=lines)  
lines_gdf.set_crs(epsg=4326, inplace=True).to_crs(epsg=32640)  # Set the CRS for lines  

# Save the lines to a new shapefile  
output_lines_shapefile_path = r'D:\AI\imageProccessingAI\shp\shp\created_lines188.shp'  
lines_gdf.to_file(output_lines_shapefile_path, driver='ESRI Shapefile')  

print("Successfully added parcel names. Output saved to detected_objects_with_parcelnames.csv.")  
print("Selected parcels saved to ghataat_selected.shp.")  
print("Created lines saved to created_lines.shp.")