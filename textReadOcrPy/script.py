# For example, on Windows it might be:  
 pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' 
# Update this path if you're on Linux or Macdef 
extract_text_from_image(image_path):  
 # Load the image 
 image = cv2.imread(image_path)  
 # Check if the image was loaded correctly if image is None:  
 print("Error: Could not open or find the image.")  
 return # Convert to grayscale 
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

 # Apply GaussianBlur to smooth the image 
 blurred = cv2.GaussianBlur(gray, (5,5),0)  

 # Use adaptive thresholding to create a binary image 
 binary = cv2.adaptiveThreshold(blurred,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)  

 # Find contours in the binary image 
 contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

 # Loop through the contours and extract text from each detected board for contour in contours:  
 # Get the bounding box for each contour 
 x, y, w, h = cv2.boundingRect(contour)  
 # Filter contours based on size (you can adjust these values)  
 if w <50 or h <15: 
 # Change these values to suit your requirements continue 
 # Crop the detected area from the image 
 roi = image[y:y+h, x:x+w]  

 # Convert ROI to string using Tesseract 
 extracted_text = pytesseract.image_to_string(roi, lang='eng') 
 # Change 'eng' to 'fas' for Persian 
 # Draw a rectangle around the detected area 
 cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0),2)  

 # Print the extracted text
 print(f"Detected text: {extracted_text.strip()}")  

 # Show the image with detected rectangles 
 cv2.imshow("Detected Boards", image)  
 cv2.waitKey(0)  
 cv2.destroyAllWindows()  

def main():  
 # Path to your image file 
 image_path = 'path_to_your_image.jpg' 
 # Change this to your actual image path 
 # Extract text from the image 
 extract_text_from_image(image_path)  

if __name__ == "__main__":  
 main()