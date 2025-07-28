#pip install opencv-python
#pip install opencv-python-headless
#pip install pillow
#pip install pytesseract
#brew install tesseract


import cv2  
import pytesseract  

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

# Load the image  
image = cv2.imread(r'D:\AI\imageProccessing\1.jpg')  

# Convert to grayscale  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

# Denoising the image  
denoised = cv2.GaussianBlur(gray, (5, 5), 0)  

# Binarization  
_, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY)  

# Resize the image  
height, width = binary.shape  
resized_image = cv2.resize(binary, (width * 2, height * 2))  

# Recognize Persian text  
custom_config = r'-l fas --oem 3 --psm 6'  # Adjust the psm (Page Segmentation Mode) if necessary  
#custom_config = r'--oem 3 --psm 6'  # Adjust the psm (Page Segmentation Mode) if necessary  
text = pytesseract.image_to_string(resized_image, config=custom_config)  

# Print recognized text  
print("متن شناسایی شده:", text)  

# Save to file  
with open(r'D:\AI\imageProccessing\result.txt', 'w', encoding='utf-8') as file:  
    file.write(text)  

print("نتیجه در فایل 'result.txt' ذخیره شد.")

# نمایش تصویر (در صورت نیاز) 

cv2.imshow('Detected Sign', image)  
cv2.waitKey(0)  
cv2.destroyAllWindows()
