import easyocr  
import cv2  
import os  
import numpy as np  
import pandas as pd  
import csv  
from ultralytics import YOLO  

image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'motor'))  
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_outMultiTask'))  # Output directory  
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_models/yolo'))  

# Load the YOLOv5 or YOLOv8 model  
module_path = os.path.join(module_dir, "yolov5x6u.pt")  
model = YOLO(module_path)  

def extract_text_from_image(image):  
    reader = easyocr.Reader(['fa'])  
    result = reader.readtext(image, detail=1, paragraph=True)  
    texts = []  
    for item in result:  
        if len(item) == 3:  
            bbox, text, prob = item  
            texts.append((bbox, text, prob))  
        elif len(item) == 2:  
            bbox, text = item  
            texts.append((bbox, text, 0.0))  
    return texts  

def detect_and_blur_objects(image, texts):  
    results = model(image)  
    car_bboxes = []  

    for result in results:  
        if hasattr(result, 'boxes') and result.boxes is not None:  
            boxes = result.boxes  
            for box in boxes:  
                if box.xyxy.size(0) > 0:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    cls = int(box.cls[0].item())  

                    if cls == 2 or cls == 7:  # Assuming class index for 'car'  
                        car_bboxes.append((x1, y1, x2, y2))  

                    if cls == 0:  # Assuming class index for 'person'  
                        person_roi = image[y1:y2, x1:x2]  
                        if person_roi.size > 0:  
                            image[y1:y2, x1:x2] = cv2.GaussianBlur(person_roi, (45, 45), 30)  

    valid_texts = []  
    for bbox, text, prob in texts:  
        top_left = tuple(map(int, bbox[0]))  
        bottom_right = tuple(map(int, bbox[2]))  
        text_x1, text_y1 = top_left  
        text_x2, text_y2 = bottom_right  

        overlap = False  
        for car_bbox in car_bboxes:  
            car_x1, car_y1, car_x2, car_y2 = car_bbox  

            if not (text_x2 < car_x1 or text_x1 > car_x2 or text_y2 < car_y1 or text_y1 > car_y2):  
                image[text_y1:text_y2, text_x1:text_x2] = cv2.GaussianBlur(image[text_y1:text_y2, text_x1:text_x2], (45, 45), 30)  
                overlap = True  
                break  

        if not overlap:  
            center_point = ((text_x1 + text_x2) // 2, (text_y1 + text_y2) // 2)  
            valid_texts.append((bbox, text, center_point))  

    return image, valid_texts  

def save_texts_to_csv(image_name, valid_texts, csv_writer):  
    for bbox, text, center_point in valid_texts:  
        top_left = tuple(map(int, bbox[0]))  
        bottom_right = tuple(map(int, bbox[2]))  

        csv_writer.writerow([  
            image_name,  
            f"{top_left[0]},{top_left[1]},{bottom_right[0]},{bottom_right[1]}",  
            f"{center_point[0]},{center_point[1]}",  
            text  
        ])  

def process_images_in_folder(input_folder, output_folder):  
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  
    csv_file_path = os.path.join(output_folder, 'detected_texts.csv')  

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:  
        csv_writer = csv.writer(csv_file)  
        csv_writer.writerow(["file_name", "bbox", "center_coordinates", "detected_text"])  

        for image_name in image_files:  
            image_path = os.path.join(input_folder, image_name)  
            image = cv2.imread(image_path)  

            if image is None:  
                print(f"Error loading image at path: {image_path}")  
                continue  

            detected_texts = extract_text_from_image(image)  
            final_image, valid_texts = detect_and_blur_objects(image, detected_texts)  

            # Draw rectangles and save the output image  
            for bbox, text, center_point in valid_texts:  
                top_left = tuple(map(int, bbox[0]))  
                bottom_right = tuple(map(int, bbox[2]))  
                cv2.rectangle(final_image, top_left, bottom_right, (0, 255, 0), 2)  

            output_image_name = f"output_{image_name}"  # Adding prefix to the output image name  
            output_image_path = os.path.join(output_folder, output_image_name)  
            cv2.imwrite(output_image_path, final_image)  
            print(f"Output image saved to: {output_image_path}")  

            # Save valid texts to the CSV file  
            save_texts_to_csv(image_name, valid_texts, csv_writer)  

def main():  
    # Specify the input images folder and the output folder  
    input_folder = image_dir  # Input directory with images  
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  # Create output directory if it does not exist  

    process_images_in_folder(input_folder, output_dir)  

if __name__ == "__main__":  
    main()