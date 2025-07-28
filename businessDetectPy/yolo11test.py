import cv2  
import matplotlib.pyplot as plt  
from ultralytics import YOLO  
import os  

image_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '_img'))
module_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '_models\yolo'))

print(f"input url is: {image_dir}")
# Load the pretrained YOLOv5 or YOLOv8 model  
module_path = os.path.join(module_dir, "yolov5x6u.pt")
model = YOLO(module_path)  # only coco 60 cls but deep  
#model = YOLO("yolov8x6-oiv7.pt")  # gooooood 600 cls 
#model = YOLO("yolov8x-worldv2.pt")  # Use yolov5s.pt, yolov5m.pt, etc., based on your needs  
#model = YOLO("yolov3-sppu.pt")  # Use yolov5s.pt, yolov5m.pt, etc., based on your needs  
#model = YOLO("FastSAM-s.pt")  # Use yolov5s.pt, yolov5m.pt, etc., based on your needs  
#model = YOLO("yolo11x")  # Use yolov5s.pt, yolov5m.pt, etc., based on your needs  
#model = YOLO("yolo11x-cls.yaml").load("yolo11x-cls.pt")
# Load the image  
image_name = "GS__1444.JPG"  # Change this to your image file  
image_path = os.path.join(image_dir, image_name)

image = cv2.imread(image_path)  
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB color format  
#image_resized = cv2.resize(image, (640, 640))  # Resize image   

# Perform object detection using the resized image  
#results = model(image_resized)
# Perform object detection  
results = model(image)  

# Process results  
for result in results:  
    print(result)  # Print the result object to understand its structure  
    if hasattr(result, 'boxes') and result.boxes is not None:  # Ensure boxes exist  
        boxes = result.boxes  # Get the detected bounding boxes  
        for box in boxes:  
            # Ensure that there are valid box coordinates  
            if box.xyxy.size(0) > 0:  
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates  
                conf = box.conf[0].item()  # Confidence score  
                cls = int(box.cls[0].item())  # Class index  
                
                # Draw bounding boxes and labels  
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  
                label = f"Class: {cls}, Conf: {conf:.2f}"  # Label format  
                cv2.putText(image, label, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  
            else:  
                print("No bounding box coordinates available.")  
    else:  
        print("No boxes detected.")

# Save or display the output image  
image_name = "detect_cars.jpg"
image_path = os.path.join(image_dir, image_name)
 
cv2.imwrite(image_path, image)  
print(f"Output saved to: {image_path}")  

'''
from ultralytics import YOLO  
import cv2  
import os  
import requests  
import numpy as np  

# Load the pretrained YOLO model  
model = YOLO("yolo11n.pt")  

# Path to the image  
img_path = r"D:\AI\imageProccessing\pano_app_test\py\img\GS__1430.JPG"  

# Check if the file exists  
if not os.path.isfile(img_path):  
    raise FileNotFoundError(f"Image file not found: {img_path}")  

# Load the image  
image = cv2.imread(img_path)  
if image is None:  
    raise ValueError("Image cannot be loaded. Check the file format or path.")  

# Perform object detection on the loaded image  
results = model(image)

'''
