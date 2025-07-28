import numpy as np  
from skimage import io, filters  

# Load the image  
image = io.imread(r'D:\AI\imageProccessing\pano_app_test\py\img\GS__1676.jpg')  

# Apply Unsharp Masking (adjust the parameters as needed)  
sharpened_image = filters.unsharp_mask(image, radius=1, amount=1.5)  

# Convert to uint8  
sharpened_image = (sharpened_image * 255).astype(np.uint8)  

# Save the output  
io.imsave(r'D:\AI\imageProccessing\pano_app_test\py\img\enhanced_cv.jpg', sharpened_image)

# Paths to input and output images
#input_image_path = r'D:\AI\imageProccessing\pano_app_test\py\img\GS__1676.jpg'
#output_image_path = r'D:\AI\imageProccessing\pano_app_test\py\img\enhanced_cv.jpg'
#model_path = r'D:\AI\imageProccessing\pano_app_test\py\models\LapSRN_x2.pb'
