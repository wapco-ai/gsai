import paddleocr
import cv2
import os
import csv

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(2, 2))
    gray = clahe.apply(gray)
    
    gray = cv2.medianBlur(gray, 5)
    
    return gray

def extract_text_from_image(image_path, ocr):
    image = preprocess_image(image_path)
    colored_image = cv2.imread(image_path)
    result = ocr.ocr(image, cls=True)
    detections = []

    if result and len(result) > 0 and result[0] is not None:
        for line in result[0]:
            text, bbox, prob = line[1][0], line[0], line[1][1]
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            center_x = (top_left[0] + bottom_right[0]) // 2
            center_y = (top_left[1] + bottom_right[1]) // 2
            center_coordinates = (center_x, center_y)
            detections.append((os.path.basename(image_path), text, prob, top_left, bottom_right, center_coordinates))
            
            cv2.rectangle(colored_image, top_left, bottom_right, (0, 255, 0), 2)
    
    return detections, colored_image

def main():
    image_folder = r'D:\AI\imageProccessing\pano_app_test\motor\unused'  # Folder containing the images
    image_out_folder = r'D:\AI\imageProccessing\pano_app_test\motor_out_paddle'  # Folder containing the images
    output_csv = r'D:\AI\imageProccessing\pano_app_test\result_paddle.csv'
    
    ocr = paddleocr.PaddleOCR(lang='fa',use_angle_cls=True)  # Use 'fa' for Persian
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name', 'detected_text', 'confidence', 'top_left', 'bottom_right', 'center_coordinates'])
        
        for filename in os.listdir(image_folder):
            if filename.endswith('.jpg') or filename.endswith('.JPG'):  # Adjust for your file types
                image_path = os.path.join(image_folder, filename)
                print(f"Processing image: {image_path}")
                detections, processed_image = extract_text_from_image(image_path, ocr)
                
                if not detections:
                    print(f"No text detected in {image_path}")
                for detection in detections:
                    writer.writerow(detection)
                
                output_image_path = os.path.join(image_out_folder, f'processed_{filename}')
                cv2.imwrite(output_image_path, processed_image)
                print(f"Saved processed image: {output_image_path}")

if __name__ == "__main__":
    main()
