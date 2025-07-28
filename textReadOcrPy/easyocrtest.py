import easyocr  
import cv2  
import matplotlib.pyplot as plt  
import os  
import numpy as np  

# Set to a non-GUI backend first if you have issues with the display  
# import matplotlib  
# matplotlib.use('Agg')  # Uncomment this to use a non-GUI backend  
image_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '_img'))
script_path = os.path.abspath(__file__)  # Get the absolute path of the script  
script_dir = os.path.split(script_path)[0]  # Get the directory of the script  

def extract_text_from_image(image_path):  
    # Load the image  
    image = cv2.imread(image_path)  
    
    if image is None:  
        print(f"Error loading image at path: {image_path}")  
        return  

    # Initialize the OCR reader  
    reader = easyocr.Reader(['fa', 'en'])  

    # Detect text in the image  
    result = reader.readtext(image, detail = 1, paragraph=True)

    # Prepare the output file for detected text  
    filename = 'result.txt'  
    file_path = os.path.join(image_dir, filename)  

    # Open a file to save the detected text  
    with open(file_path, 'w', encoding='utf-8') as f:  
        for item in result:  
            if len(item) == 3:  # Ensure there are three values to unpack  
                bbox, text, prob = item  
                f.write(f"Detected text: {text} - Confidence: {prob:.2f}\n")  
                
                # Draw a rectangle around the detected text  
                top_left = tuple(map(int, bbox[0]))  
                bottom_right = tuple(map(int, bbox[2]))  
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2) 
            elif len(item) == 2:  # Ensure there are three values to unpack  
                bbox, text = item  
                f.write(f"Detected text: {text} - Confidence: {0:.2f}\n")  
        
        # Draw a rectangle around each detected text  
                top_left = tuple(map(int, bbox[0]))  
                bottom_right = tuple(map(int, bbox[2]))  
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  
            else:  
                print(f"Unexpected result format: {item}")

    # Convert the image from BGR to RGB (as OpenCV uses BGR format)  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    # Display the image using Matplotlib  
    '''
    plt.figure(figsize=(10, 8))  
    plt.imshow(image_rgb)  
    plt.title("Detected Text")  
    plt.axis("off")  # Hide axes  
    plt.show()  # Show the image  
    '''
    # Save the result image if needed  
    image_name = 'detected.jpg'  
    image_path = os.path.join(image_dir, image_name)  
    plt.imsave(image_path, image_rgb.astype(np.uint8))  # Save the image  

def main():  
    image_name = 'GS__16766.JPG'  # Path to your actual image  
    image_path = os.path.join(image_dir, image_name)  
    extract_text_from_image(image_path)  

if __name__ == "__main__":  
    main()