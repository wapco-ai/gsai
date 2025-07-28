#pip install opencv-python
#pip install opencv-python-headless
#pip install pillow
#pip install pytesseract
#brew install tesseract


import cv2  
import pytesseract  

# بارگذاری تصویر  
image = cv2.imread('D:\AI\imageProccessing\Interior-view-of-Ferdowsi-Shopping-Center.jpg')  

# تبدیل تصویر به خاکستری  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

# استفاده از Tesseract برای شناسایی متن  
#custom_config = r'--oem 3 --psm 6'  english only
custom_config = r'-l fas --oem 3 --psm 6' 
text = pytesseract.image_to_string(gray, config=custom_config)  

# چاپ متن شناسایی شده  
print("شناسایی متن تابلو:", text)  


# Save the recognized text to a file  
with open(r'D:\AI\imageProccessing\result.txt', 'w', encoding='utf-8') as file:  
    file.write(text)  

print("نتیجه در فایل 'D:\AI\imageProccessing\result.txt' ذخیره شد.") 

# نمایش تصویر (در صورت نیاز) 
''' 
cv2.imshow('Detected Sign', image)  
cv2.waitKey(0)  
cv2.destroyAllWindows()
'''