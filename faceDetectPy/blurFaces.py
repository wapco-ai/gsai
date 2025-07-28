import cv2  

# Load the image  
image = cv2.imread(r'D:\AI\imageProccessing\pano_app_test\motor\unused\GS__1427.JPG')  

# Load the pre-trained Haar Cascade Classifier for face detection  
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
face_cascade = cv2.CascadeClassifier(r'D:\AI\imageProccessing\pano_app_test\py\models\lbpcascade_frontalcatface.xml')  

# Convert image to grayscale for face detection  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

# Detect faces in the image  
faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)  

# Blur each detected face  
for (x, y, w, h) in faces:  
    face_region = image[y:y+h, x:x+w]  
    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)  
    image[y:y+h, x:x+w] = blurred_face  

# Save the edited image  
cv2.imwrite(r'D:\AI\imageProccessing\pano_app_test\motor\unused\blured.JPG', image)