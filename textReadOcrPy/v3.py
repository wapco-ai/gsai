#pip install opencv-python
#pip install opencv-python-headless
#pip install pillow
#pip install pytesseract
#brew install tesseract


from PIL import Image  
import pytesseract 
import matplotlib.pyplot as plt  
 

# Load the image  
image = Image.open(r'D:\AI\imageProccessing\1.jpg')  

# Convert to grayscale  
image = image.convert('L') 
#threshold_image = image.point(lambda x: 0 if x < 128 else 255, '1')

#tesseract threshold_image output --lang fas
#tesseract threshold_image output --psm 6  # Assume a uniform block of text 

# Optionally, apply other preprocessing steps here  

# Run Tesseract OCR  
text = pytesseract.image_to_string(image, lang='fas')  # lang can be changed as needed  

print(text)


# Save to file  
with open(r'D:\AI\imageProccessing\result.txt', 'w', encoding='utf-8') as file:  
    file.write(text)  

print("نتیجه در فایل 'result.txt' ذخیره شد.")

# Display the image  
plt.imshow(image)  
plt.axis('off')  # Hide axes  
plt.show() 
