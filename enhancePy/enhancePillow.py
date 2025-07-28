from PIL import Image

def enhance_resolution(image_path, output_path, scale_factor=2):
    # Load the image
    image = Image.open(image_path)
    
    # Calculate new dimensions
    new_width = image.width * scale_factor
    new_height = image.height * scale_factor
    
    # Resize the image
    enhanced_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Save the enhanced image
    enhanced_image.save(output_path)
    print(f"Enhanced image saved as '{output_path}'")

# Paths to input and output images
input_image_path = 'D:\AI\imageProccessing\pano_app_test\py\img\GS__1676.jpg'
output_image_path = 'D:\AI\imageProccessing\pano_app_test\py\img\enhanced_pill.jpg'

enhance_resolution(input_image_path, output_image_path)

