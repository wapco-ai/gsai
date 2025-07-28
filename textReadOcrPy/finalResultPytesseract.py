import requests  
from PIL import Image  
import pytesseract  
import cv2  
import numpy as np  
from io import BytesIO  
import matplotlib.pyplot as plt  


# URL of the image  
#url = 'http://89.42.210.15/dl/Caspian-Mode.jpg'  

# Download the image  
#response = requests.get(url)  
#image = Image.open(BytesIO(response.content))  
image = Image.open(r'D:\AI\imageProccessing\pano_app_test\GS__1430.jpg')  


# Convert to grayscale  
image = image.convert('L')  

# Apply binary thresholding to improve OCR results  
threshold = 100  
image_binary = image.point(lambda x: 0 if x < threshold else 255, '1')  

# Convert PIL image to numpy array  
opencv_image = np.array(image_binary)  

# Ensure the image is in a numeric format suitable for OpenCV  
opencv_image = opencv_image.astype(np.uint8) * 255  # Convert boolean to uint8  

# Apply morphological transformations  
kernel = np.ones((3, 3), np.uint8)  
opencv_image = cv2.morphologyEx(opencv_image, cv2.MORPH_CLOSE, kernel)  

# Optionally, apply Gaussian blur (can help in some cases)  
opencv_image = cv2.GaussianBlur(opencv_image, (3, 3), 0)  

# Optional: Use adaptive thresholding if needed  
# opencv_image = cv2.adaptiveThreshold(opencv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
#                                      cv2.THRESH_BINARY, 11, 2)  

# Run Tesseract OCR  
text = pytesseract.image_to_string(opencv_image, config='-l fas --psm 6')  

# Save to file  
with open(r'D:\AI\imageProccessing\pano_app_test\result.txt', 'w', encoding='utf-8') as file:  
    file.write(text)  

print("نتیجه در فایل 'result.txt' ذخیره شد.")

# Display the image  
plt.imshow(image)  
plt.axis('off')  # Hide axes  
plt.show() 
