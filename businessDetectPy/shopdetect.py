import os
import cv2
from ultralytics import YOLOv5

# Define the image path
img_path = 'path_to_your_image.jpg'
img_path = os.path.abspath(img_path)  # Ensure the path is absolute

# Verify the image file exists
if not os.path.isfile(img_path):
    print(f"Image file not found: {img_path}")
    exit()

# Load the pre-trained model
model = YOLOv5('path_to_yolov5_weights.pt')

# Read the image
img = cv2.imread(img_path)

# Verify image is loaded correctly
if img is None:
    print(f"Failed to load image: {img_path}")
    exit()

# Run the model on the image
results = model(img_path)

# Display the results
results.print()  # Print results to the console
results.show()   # Display results in a new window
results.save()   # Save results to a file
'''
import torch  
import cv2  
import matplotlib.pyplot as plt  

# Load the YOLOv5 model (you can use any version)  
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='github')  # Load from GitHub  

# Load an image from a file  
img_path = r'D:\AI\imageProccessing\pano_app_test\py\img\6vQWGoJxAJiybt-fd8c097c269d4a1da9eb859215eff7e6.jpg'  # Change to your image file path  
img = cv2.imread(img_path)  

# Perform inference  
results = model(img)  

# Print results  
results.print()  # Prints the results  
results.show()   # Displays the image with detections  

# Save results to a file  
results.save()   # Saves results to 'runs/detect/exp/'  

# Optionally, you can get the bounding boxes and class names  
boxes = results.xyxy[0].numpy()  # Get the detections  
for box in boxes:  
    x1, y1, x2, y2, conf, cls = box  
    print(f"Detected class: {model.names[int(cls)]}, Confidence: {conf:.2f}, Coordinates: ({x1}, {y1}), ({x2}, {y2})")
'''