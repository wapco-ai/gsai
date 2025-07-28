import cv2
import numpy as np

# Load the pre-trained EAST text detector model
net = cv2.dnn.readNet(r'D:\AI\imageProccessing\pano_app_test\py\models\opencv\frozen_east_text_detection.pb')

# Define the function to decode the output layers
def decode_predictions(scores, geometry, scoreThresh):  
    detections = []  
    confidences = []  

    for y in range(0, scores.shape[2]):  
        scoresData = scores[0, 0, y]  
        x0Data = geometry[0, 0, y]  
        x1Data = geometry[0, 1, y]  
        x2Data = geometry[0, 2, y]  
        x3Data = geometry[0, 3, y]  
        anglesData = geometry[0, 4, y]  

        for x in range(0, scores.shape[3]):  
            score = scoresData[x]  
            if score < scoreThresh:  
                continue  

            offsetX, offsetY = x * 4.0, y * 4.0  
            angle = anglesData[x]  
            cosA, sinA = np.cos(angle), np.sin(angle)  
            h = x0Data[x] + x2Data[x]  
            w = x1Data[x] + x3Data[x]  
            
            # Calculate the center for the bounding box  
            centerX = int(offsetX + (cosA * x1Data[x]) + (sinA * x2Data[x]))  
            centerY = int(offsetY - (sinA * x1Data[x]) + (cosA * x2Data[x]))  
            
            # Create the rotated box information for NMS  
            detections.append((centerX, centerY, w, h, angle * 180.0 / np.pi))  # angle in degrees  
            confidences.append(float(score))  

    return detections, confidences  

# Load the image  
image = cv2.imread(r"D:\AI\imageProccessing\pano_app_test\py\img\4.JPG")  
orig = image.copy()  
(H, W) = image.shape[:2]  

# Set the new width and height and determine the ratio  
(newW, newH) = (320, 320)  
rW = W / float(newW)  
rH = H / float(newH)  

# Resize the image  
image = cv2.resize(image, (newW, newH))  
(H, W) = image.shape[:2]  

# Define the output layer names  
layerNames = [  
    "feature_fusion/Conv_7/Sigmoid",  
    "feature_fusion/concat_3"  
]  

# Blob creation and model forward pass  
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)  
net.setInput(blob)  
(scores, geometry) = net.forward(layerNames)  

# Decode the predictions  
detections, confidences = decode_predictions(scores, geometry, 0.5)  

# Apply non-maxima suppression to suppress weak, overlapping bounding boxes  
if len(detections) > 0:  # Ensure you have detections to proceed  
    indices = cv2.dnn.NMSBoxesRotated(detections, confidences, 0.5, 0.4)  

# Draw the bounding boxes on the image  
if len(indices) > 0:  
    for i in indices.flatten():  
        (centerX, centerY, width, height, angle) = detections[i]  
        # Convert back to original image dimensions  
        centerX = int(centerX * rW)  
        centerY = int(centerY * rH)  
        width = int(width * rW)  
        height = int(height * rH)  
        cv2.rectangle(orig, (centerX, centerY), (centerX + width, centerY + height), (0, 255, 0), 2)  

# Save and display the result  
output_path = r"D:\AI\imageProccessing\pano_app_test\py\img\detected-sign.JPG"  
cv2.imwrite(output_path, orig)  
cv2.imshow('Shop Signs Detected', orig)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  
print(f"Detected image saved as '{output_path}'")