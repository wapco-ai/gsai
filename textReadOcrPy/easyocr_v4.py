import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

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
    #cv2.imwrite(r'D:\AI\imageProccessing\pano_app_test\py\img\1451-enhanced.jpg', resized_image)
    
    return image

def enhanced_preprocess_image(image_path):  
    # Load the image  
    image = cv2.imread(image_path)  

    # Convert to Grayscale  
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    # Denoising  
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)  

    # Histogram Equalization  
    equalized_image = cv2.equalizeHist(denoised_image)  

    # Adaptive Thresholding  
    binary_image = cv2.adaptiveThreshold(equalized_image, 255,   
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   
                                         cv2.THRESH_BINARY_INV, 11, 2)  

    # Morphological Operations  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)  

    # Dilation  
    dilated_image = cv2.dilate(closed_image, kernel, iterations=1)  

    # Optionally Deskew (already provided in your previous code)  
    coords = np.column_stack(np.where(dilated_image > 0))  
    angle = cv2.minAreaRect(coords)[-1]  
   
    if angle < -45:  
        angle = -(90 + angle)  
    else:  
        angle = -angle  

    (h, w) = dilated_image.shape[:2]  
    center = (w // 2, h // 2)  
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  
    rotated_image = cv2.warpAffine(dilated_image, M, (w, h), flags=cv2.INTER_CUBIC)  

    return image  
    
def extract_text_from_image(image_path):  
    # Preprocess the image  
    #preprocessed_image = preprocess_image(image_path)  
    preprocessed_image = enhanced_preprocess_image(image_path)  

    # Initialize the reader  
    reader = easyocr.Reader(['fa', 'en'])  # Use 'fa' for Persian  

    # Detect text in the preprocessed image  
    result = reader.readtext(preprocessed_image)  

    # Read the original colored image to draw rectangles  
    colored_image = cv2.imread(image_path)  

    # Open a file to save the detected text  
    with open(r'D:\AI\imageProccessing\pano_app_test\py\img\result.txt', 'w', encoding='utf-8') as f:  
        for (bbox, text, prob) in result:  
            # Draw a rectangle around each detected text  
            top_left = tuple(map(int, bbox[0]))  
            bottom_right = tuple(map(int, bbox[2]))  

            # Write detected text and coordinates to file  
            f.write(f"Detected text: {text} - Confidence: {prob:.2f} - Coordinates: {top_left}, {bottom_right}\n")  

            # Draw rectangles on the colored image  
            cv2.rectangle(colored_image, top_left, bottom_right, (0, 255, 0), 2)  

    # Convert colored image from BGR to RGB (OpenCV uses BGR by default)  
    image_rgb = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)  

    # Display or save the result image  
    plt.imsave(r'D:\AI\imageProccessing\pano_app_test\py\img\detect.jpg', image_rgb)

def main():
    image_path = r'D:\AI\imageProccessing\pano_app_test\motor\GS__1756.jpg'  # Change this to your actual image path
    extract_text_from_image(image_path)

if __name__ == "__main__":
    main()
