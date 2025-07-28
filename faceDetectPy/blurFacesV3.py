import cv2
from mtcnn.mtcnn import MTCNN

def detect_and_blur_faces(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Initialize the face detector
    detector = MTCNN()
    
    # Detect faces in the image
    faces = detector.detect_faces(image)
    
    # Blur each detected face
    for face in faces:
        x, y, width, height = face['box']
        x, y = abs(x), abs(y)
        roi = image[y:y+height, x:x+width]
        blurred_roi = cv2.GaussianBlur(roi, (23, 23), 30)
        image[y:y+height, x:x+width] = blurred_roi
    
    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Faces blurred and image saved as '{output_path}'")

# Paths to input and output images
input_image_path = r'D:\AI\imageProccessing\pano_app_test\motor\unused\GS__1427.JPG'
output_image_path = r'D:\AI\imageProccessing\pano_app_test\motor\unused\blur.JPG'

detect_and_blur_faces(input_image_path, output_image_path)
