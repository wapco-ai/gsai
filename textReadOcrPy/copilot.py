import pytesseract
from PIL import Image
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Remove noise
    gray = cv2.medianBlur(gray, 5)
    # Binarize image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text_from_image(image_path):
    image = preprocess_image(image_path)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 50 or h < 15:
            continue
        roi = image[y:y+h, x:x+w]
        extracted_text = pytesseract.image_to_string(roi, lang='fas')  # 'eng' for English
        print(f"Detected text: {extracted_text.strip()}")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Resize window to fit the image
    cv2.namedWindow("Detected Text", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Text", image)
    cv2.resizeWindow("Detected Text", 800, 600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = r'D:\AI\imageProccessing\pano_app_test\GS__1430.jpg'
    extract_text_from_image(image_path)

if __name__ == "__main__":
    main()
