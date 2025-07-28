import json
import math

def calculate_azimuth(lat1, lon1, lat2, lon2):
    """ Calculate the azimuth between two points on the earth (specified in decimal degrees) """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    d_lon = lon2 - lon1

    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)

    azimuth = math.atan2(x, y)
    azimuth = math.degrees(azimuth)
    azimuth = (azimuth + 360) % 360  # Normalize to 0-360 degrees
    azimuth = 180 - round(azimuth)
    
    return azimuth

def update_json_with_azimuth(json_data):
    for node in json_data:
        if node['links']:
            next_node_id = node['links'][0]['nodeId']
            next_node = next((item for item in json_data if item["id"] == next_node_id), None)
            
            if next_node:
                lat1, lon1 = float(node['gps'][1]), float(node['gps'][0])
                lat2, lon2 = float(next_node['gps'][1]), float(next_node['gps'][0])
                
                azimuth = calculate_azimuth(lat1, lon1, lat2, lon2)
                node['sphereCorrection']['pan'] = f"{azimuth}deg"
    
    return json_data

def main():
    # Read JSON data from file
    with open(r'D:\AI\imageProccessing\pano_app_test\motor\shp\nodes.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Update JSON data with azimuth values
    updated_json = update_json_with_azimuth(json_data)

    # Save updated JSON data to another file
    with open(r'D:\AI\imageProccessing\pano_app_test\py\genereate_tour_json\nodesA.json', 'w', encoding='utf-8') as file:
        json.dump(updated_json, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
