#pip install pytesseract opencv-python Pillow requests numpy
import requests  
from PIL import Image  
import pytesseract  
import cv2  
import numpy as np  
from io import BytesIO  
import matplotlib.pyplot as plt  


# Set the Tesseract executable path if necessary  
# For example, on Windows, it might be:  
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

image = Image.open(r'D:\AI\imageProccessing\pano_app_test\GS__1430.jpg')

# Convert the image to OpenCV format  
opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  

# Convert to grayscale  
gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)  

# Enhance image: increase contrast and apply a Gaussian blur  
gray_image = cv2.convertScaleAbs(gray_image, alpha=1.5, beta=0)  # Adjust alpha for contrast  
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  

# Use Otsu's thresholding for binarization  
_, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

# Dilation to help connect text components  
kernel = np.ones((3, 3), np.uint8)  
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)  

# Find contours in the dilated image  
contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

# Filter based on area to detect potential signs  
signs_with_text = []  
for contour in contours:  
    area = cv2.contourArea(contour)  
    if area > 1000:  # Increase to filter for larger signs  
        x, y, w, h = cv2.boundingRect(contour)  
        
        # Extract the ROI  
        roi = opencv_image[y:y + h, x:x + w]  
        
        # Prepare ROI for OCR  
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
        _, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

        # Run Tesseract OCR  
        custom_config = r'--psm 6'  # Adjust as necessary  
        text = pytesseract.image_to_string(roi_binary, lang='eng', config=custom_config)  

        # Check if any text was detected  
        if text.strip():  
            signs_with_text.append((x, y, w, h, text.strip()))  
            cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  

# Output detected signs with text  
for (x, y, w, h, text) in signs_with_text:  
    print(f"Detected text: '{text}' at position: ({x}, {y}, {w}, {h})")  


# Show the original image with detected signs outlined
'''  
cv2.imshow("Detected Signs", opencv_image)  
cv2.waitKey(0)  
cv2.destroyAllWindows()
'''
# Display the image  
plt.imshow(opencv_image)  
plt.axis('off')  # Hide axes  
plt.show() 