import cv2  
import numpy as np  
import easyocr  
import os  
os.environ['OMP_NUM_THREADS'] = '1' 


def detect_and_blur_license_plates(image_path, output_path):  
    # Load the image  
    image = cv2.imread(image_path)  
    
    if image is None:  
        raise Exception("Could not read the image.")  

    # Initialize EasyOCR reader for Farsi and optionally English  
    reader = easyocr.Reader(['fa', 'en'], gpu=True)  # You can specify other languages if needed  

    # Run EasyOCR to detect text in the image  
    results = reader.readtext(image)  

    for (bbox, text, prob) in results:  
        # Extract the bounding box coordinates  
        (top_left, top_right, bottom_right, bottom_left) = bbox  
        top_left = tuple(map(int, top_left))  
        bottom_right = tuple(map(int, bottom_right))  
        
        # Assuming license plates have a high confidence level  
        if prob > 0.7:  # You can adjust this threshold  
            # Get the region of interest (ROI)  
            roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]  
            
            # Apply Gaussian blur to the ROI  
            blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)  
            
            # Replace the ROI in the original image with the blurred version  
            image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi  

    # Save the output image  
    cv2.imwrite(output_path, image)  
    print(f"Processed image saved to: {output_path}")  


# Example usage  
input_image_path = r'D:\AI\imageProccessing\pano_app_test\GS__1535.jpg'  # Path to input image  
output_image_path = r'D:\AI\imageProccessing\pano_app_test\GS__1535_blurred_license_plate.jpg'  # Path to save output image  

detect_and_blur_license_plates(input_image_path, output_image_path)