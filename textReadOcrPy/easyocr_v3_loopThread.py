import easyocr
import cv2
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(2, 2))
    gray = clahe.apply(gray)
    
    #gray = cv2.medianBlur(gray, 5)
    
    return gray

def extract_text_from_image(image_path, reader):
    image = preprocess_image(image_path)
    colored_image = cv2.imread(image_path)

    result = reader.readtext(image)
    detections = []

    for (bbox, text, prob) in result:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        detections.append((os.path.basename(image_path), text, prob, top_left, bottom_right))
        
        cv2.rectangle(colored_image, top_left, bottom_right, (0, 255, 0), 2)
    
    return detections, colored_image

def process_image(filename, image_folder, reader):
    image_path = os.path.join(image_folder, filename)
    detections, processed_image = extract_text_from_image(image_path, reader)
    
    output_image_path = os.path.join(image_folder, f'processed_{filename}')
    cv2.imwrite(output_image_path, processed_image)
    
    return detections

def main():
    image_folder = r'D:\AI\imageProccessing\pano_app_test\motor'  # Folder containing the images
    image_out_folder = r'D:\AI\imageProccessing\pano_app_test\motor_out2'  # Folder containing the images
    output_csv = r'D:\AI\imageProccessing\pano_app_test\result2.csv'
    
    reader = easyocr.Reader(['fa', 'en'])  # Use 'fas' for Persian

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name', 'detected_text', 'confidence', 'top_left', 'bottom_right'])
        
        filenames = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.JPG')]  # Adjust for your file types
        max_workers = min(4, len(filenames))  # Adjust the number of workers based on system capability
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_image, filename, image_folder, reader): filename for filename in filenames}
            
            for future in as_completed(futures):
                detections = future.result()
                for detection in detections:
                    writer.writerow(detection)

if __name__ == "__main__":
    main()
