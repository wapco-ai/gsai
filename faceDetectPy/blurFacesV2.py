import cv2
import numpy as np

def detect_and_blur_faces(image_path, model_path, config_path, output_path):
    # Load the pre-trained face detection model
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    
    # Prepare the image for the deep learning model
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w, endX), min(h, endY))
            
            # Extract the face ROI and apply a Gaussian blur on it
            face = image[startY:endY, startX:endX]
            face = cv2.GaussianBlur(face, (23, 23), 30)
            image[startY:endY, startX:endX] = face
    
    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Faces blurred and image saved as '{output_path}'")

# Paths to input and output images
input_image_path = r'D:\AI\imageProccessing\pano_app_test\motor\unused\GS__1427.JPG'
output_image_path = r'D:\AI\imageProccessing\pano_app_test\motor\unused\blur.JPG'
model_path = r'D:\AI\imageProccessing\pano_app_test\py\models\res10_300x300_ssd_iter_140000_fp16.caffemodel'
config_path = r'D:\AI\imageProccessing\pano_app_test\py\models\deploy.prototxt'

detect_and_blur_faces(input_image_path, model_path, config_path, output_image_path)