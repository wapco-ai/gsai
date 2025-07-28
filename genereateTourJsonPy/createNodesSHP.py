import geopandas as gpd  
import json  
from geopy.distance import distance as geopy_distance  

def create_nodes_from_shp(shp_file, base_url, link_distance=10):  
    # Load the shapefile  
    gdf = gpd.read_file(shp_file)  

    if not {'filename', 'latitude', 'longitude', 'altitude'}.issubset(gdf.columns):  
        raise ValueError("The shapefile must contain 'filename', 'latitude', 'longitude', and 'name' fields.")  

    nodes = []  
    sphereCorrection = {"pan": "-100deg"}
    for index, row in gdf.iterrows():  
        node_id = str(index + 1)  
        panorama = f"{base_url}{row['filename']}.jpg"  
        name = row['filename']  
        gps = [row['longitude'], row['latitude'], row['altitude']]  # Longitude, Latitude, Altitude  
        links = []  

        nodes.append({  
            'id': node_id,  
            'panorama': panorama,  
            'name': name,  
            'links': links,  
            'gps': gps,  
            'sphereCorrection': {"pan": "105deg"}
#            "sphereCorrection": sphereCorrection
        })  
    
     # Create links for previous and next nodes  
    for i, node in enumerate(nodes):  
        if i > 0:  # Link to previous node if it exists  
            node['links'].append({'nodeId': nodes[i - 1]['id']})  

        if i < len(nodes) - 1:  # Link to next node if it exists  
            node['links'].append({'nodeId': nodes[i + 1]['id']}) 
            
    # Link nodes that are within the defined distance  
    for i, node in enumerate(nodes):  
        current_gps = (node['gps'][1], node['gps'][0])  # (latitude, longitude) for geopy  

        for j, other_node in enumerate(nodes):  
            if i != j:  
                other_gps = (other_node['gps'][1], other_node['gps'][0])  
                dist = geopy_distance(current_gps, other_gps).meters  # Calculate distance in meters  

                if dist <= link_distance:  
                    new_link = {'nodeId': other_node['id']}  
                    if new_link not in node['links']:  # Check for duplicate link  
                        node['links'].append(new_link)  

    return nodes  

def save_nodes_to_json(nodes, output_file):  
    with open(output_file, 'w') as f:  
        json.dump(nodes, f, indent=4)  

# Example usage  
shp_file = r'D:\AI\imageProccessing\DCIM\apptest\motor\shp\taged_images.shp'  # Specify the path to your shapefile  
base_url = 'http://localhost:8000/apptest/motor/'  # Replace with your actual base URL  
output_file = r'D:\AI\imageProccessing\DCIM\apptest\motor\shp\nodes.json'  # File to save the nodes 

try:  
    nodes = create_nodes_from_shp(shp_file, base_url)  
    save_nodes_to_json(nodes, output_file)  
    print(f"Nodes saved to {output_file}")  
except Exception as e:  
    print(f"Error: {e}")