import cv2  
import numpy as np  

# Load the image  
image = cv2.imread(r'D:\AI\imageProccessing\pano_app_test\py\img\1451.jpg')  

# Convert to Grayscale  
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

# Binarization  
binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
                                     cv2.THRESH_BINARY_INV, 11, 2)  

# Noise Reduction  
blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)  

# Dilation  
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
dilated_image = cv2.dilate(blurred_image, kernel, iterations=1)  

# Deskewing (optional, based on your image orientation)  
coords = np.column_stack(np.where(dilated_image > 0))  
angle = cv2.minAreaRect(coords)[-1]  

if angle < -45:  
    angle = -(90 + angle)  
else:  
    angle = -angle  

(h, w) = dilated_image.shape[:2]  
center = (w // 2, h // 2)  
M = cv2.getRotationMatrix2D(center, angle, 1.0)  
rotated_image = cv2.warpAffine(dilated_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  

# Resize the image if needed  
resized_image = cv2.resize(rotated_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  

# Save the preprocessed image  
cv2.imwrite(r'D:\AI\imageProccessing\pano_app_test\py\img\1451-enhanced.jpg', resized_image)