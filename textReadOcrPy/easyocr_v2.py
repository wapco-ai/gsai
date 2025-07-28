import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Improve image contrast
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(2, 2))
    gray = clahe.apply(gray)
    
    # Remove noise
    gray = cv2.medianBlur(gray, 5)
    
    return gray

def extract_text_from_image(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)
    colored_image = cv2.imread(image_path)

    # Initialize the reader
    reader = easyocr.Reader(['fa', 'en'])  # Use 'fa' for Persian

    # Detect text in the image
    result = reader.readtext(image)

    # Open a file to save the detected text
    with open(r'D:\AI\imageProccessing\pano_app_test\result.txt', 'w', encoding='utf-8') as f:
        for (bbox, text, prob) in result:            
            # Draw a rectangle around each detected text
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            f.write(f"Detected text: {text} - Confidence: {prob:.2f} - Coordinates: {top_left}, {bottom_right}\n")
            
            cv2.rectangle(colored_image, top_left, bottom_right, (0, 255, 0), 2)
    
    # Convert image from BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    # Display the image with matplotlib
    #plt.imshow(image_rgb)
    #plt.title("Detected Text")
    #plt.axis("off")
    #plt.show()

    # Save the result image
    plt.imsave(r'D:\AI\imageProccessing\pano_app_test\py\img\enhanced_cv_detect.jpg', image_rgb)

def main():
    image_path = r'D:\AI\imageProccessing\pano_app_test\py\img\enhanced_cv.jpg'  # Change this to your actual image path
    extract_text_from_image(image_path)

if __name__ == "__main__":
    main()
