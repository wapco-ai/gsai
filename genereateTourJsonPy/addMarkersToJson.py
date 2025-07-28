import json
import csv
import os  

script_path = os.path.abspath(__file__)  # Get the absolute path of the script  
script_dir = os.path.split(script_path)[0]  # Get the directory of the script 

def read_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            data.append(dict(zip(headers, row)))
    return data

def update_json_with_markers(json_data, csv_data):
    marker_id = 1
    for csv_row in csv_data:
        csv_name = csv_row['file_name'].split('.')[0].lower()  # Extract base name and lower case
        csv_confidence = csv_row['confidence']
        for node in json_data:
            json_name = node['name'].lower()
            if json_name == csv_name and float(csv_confidence) > 0.1:
                marker = {
                    "id": f"marker_{marker_id}",
                    "circle": 20,
                    "position": {
                        "textureX": int(csv_row['center_coordinates'].split(',')[0].strip().replace("(", "")),
                        "textureY": int(csv_row['center_coordinates'].split(',')[1].strip().replace(")", ""))                        
                    },
                    "tooltip": f"متن: {csv_row['detected_text']} <br> دقت: {csv_row['confidence']}",
                    "svgStyle": {
                        "fill" : "rgba(0, 0, 0, 0.5)",
                        "stroke": "#ff0000",
                        "strokeWidth": "4px"
                    }
                }
                
                if "markers" not in node:
                    node["markers"] = []
                
                node["markers"].append(marker)
                marker_id += 1
    
    return json_data

def main():
    # Read JSON data from file
    with open(r'D:\AI\imageProccessing\pano_app_test\py\genereate_tour_json\nodesA.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Read CSV data
    csv_data = read_csv(r'D:\AI\imageProccessing\pano_app_test\py\genereate_tour_json\result.csv')

    # Update JSON data with markers
    updated_json = update_json_with_markers(json_data, csv_data)

    # Save updated JSON data to another file
    with open(r'D:\AI\imageProccessing\pano_app_test\py\genereate_tour_json\nodesB.json', 'w', encoding='utf-8') as file:
        json.dump(updated_json, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
