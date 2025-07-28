import json
import csv

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
        for node in json_data:
            json_name = node['name'].lower()
            if json_name == csv_name:
                marker = {
                    "id": f"text_{marker_id}",
                    "html": f"<b>{csv_row['detected_text']}</b>;",
                    "anchor": 'bottom right',
                    "scale": [0.5, 1.5],
                    "style": {
                        "maxWidth": "100px",
                        "color": "white",
                        "fontSize": "20px",
                        "fontFamily": "Helvetica, sans-serif",
                        "textAlign": "center",
                    },
                    "position": {
                        "textureY": int(csv_row['center_coordinates'].split(',')[1].strip().replace(")", "")),
                        "textureX": int(csv_row['center_coordinates'].split(',')[0].strip().replace("(", ""))
                    },
                    "tooltip": csv_row['detected_text'],
                    "tooltip": {
                        "content": csv_row['detected_text'],
                        "position": "right",
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
