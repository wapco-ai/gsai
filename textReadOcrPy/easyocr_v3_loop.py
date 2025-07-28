import easyocr
import cv2
import os
import csv

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Improve image contrast
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(2, 2))
    gray = clahe.apply(gray)
    
    # Remove noise
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
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2
        center_coordinates = (center_x, center_y)
        detections.append((os.path.basename(image_path), text, prob, top_left, bottom_right, center_coordinates))
        
        cv2.rectangle(colored_image, top_left, bottom_right, (0, 255, 0), 2)
    
    return detections, colored_image

def main():
    image_folder = r'D:\AI\imageProccessing\pano_app_test\motor'  # Folder containing the images
    image_out_folder = r'D:\AI\imageProccessing\pano_app_test\motor_out'  # Folder containing the images
    output_csv = r'D:\AI\imageProccessing\pano_app_test\result.csv'
    
    reader = easyocr.Reader(['fa', 'en'])  # Use 'fa' for Persian
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name', 'detected_text', 'confidence', 'top_left', 'bottom_right', 'center_coordinates'])
        
        for filename in os.listdir(image_folder):
            if filename.endswith('.jpg') or filename.endswith('.JPG'):  # Adjust for your file types
                image_path = os.path.join(image_folder, filename)
                print(f"Processing image: {image_path}")
                detections, processed_image = extract_text_from_image(image_path, reader)
                
                if not detections:
                    print(f"No text detected in {image_path}")

                for detection in detections:
                    writer.writerow(detection)
                
                # Save the processed image
                output_image_path = os.path.join(image_out_folder, f'processed_{filename}')
                cv2.imwrite(output_image_path, processed_image)
                print(f"Saved processed image: {output_image_path}")

if __name__ == "__main__":
    main()
