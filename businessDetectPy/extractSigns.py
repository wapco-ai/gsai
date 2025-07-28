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

image = Image.open(r'D:\AI\imageProccessing\Caspian-Mode.jpg')

# Convert the image to an OpenCV format  
opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  

# Convert to grayscale  
gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)  

# Apply Gaussian blur to reduce noise  
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  

# Use Canny edge detection to find edges in the image  
edges = cv2.Canny(blurred_image, 30, 150)  

# Find contours  
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  

# Filter based on contour area and shape to detect potential shop signs  
signs_with_text = []  
for contour in contours:  
    area = cv2.contourArea(contour)  
    if area > 1000:  # Minimum area threshold, adjust based on your image's sign size  
        x, y, w, h = cv2.boundingRect(contour)  
        
        # Extract the region of interest (ROI) for OCR  
        roi = opencv_image[y:y + h, x:x + w]  
        
        # Convert to grayscale and apply thresholding  
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
        _, roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY_INV)  

        # Run Tesseract OCR on the ROI  
        text = pytesseract.image_to_string(roi_thresh, lang='fas', config='--psm 6')  
        
        # Check if any text was detected  
        if text.strip():  # If text is detected  
            signs_with_text.append((x, y, w, h, text.strip()))  
            # Draw the bounding box around detected sign on the original image  
            cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  

# Output detected signs with text  
for (x, y, w, h, text) in signs_with_text:  
    print(f"Detected text: {text} at position: ({x}, {y}, {w}, {h})")  
# Save to file  
with open(r'D:\AI\imageProccessing\result.txt', 'w', encoding='utf-8') as file:  
    file.write(text)
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