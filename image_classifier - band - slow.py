
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import TFSegformerForSemanticSegmentation, SegformerFeatureExtractor
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configure logging for the classifier
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model and feature extractor paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# Global variables to hold the loaded model and feature extractor
model = None
feature_extractor = None

def load_model_and_feature_extractor():
    """Loads the Segformer model and feature extractor."""
    global model, feature_extractor

    if model is None or feature_extractor is None:
        logging.info("📦 Loading Segformer model and feature extractor...")
        try:
            if not os.path.exists(MODEL_DIR):
                logging.info("⬇ Model not found locally. Downloading...")
                model = TFSegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, from_pt=True)
                model.save_pretrained(MODEL_DIR)
                logging.info("✅ Model downloaded and saved.")
            else:
                logging.info("📦 Loading model from local path...")
                model = TFSegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
                logging.info("✅ Model loaded from local path.")

            feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
            logging.info("✅ Feature extractor loaded.")

        except Exception as e:
            logging.error(f"Error loading model or feature extractor: {e}")
            model = None
            feature_extractor = None
            raise

def classify_image(image_path):
    """Classifies a single image and returns the predicted segmentation mask (numpy array)."""
    if model is None or feature_extractor is None:
        load_model_and_feature_extractor()

    try:
        image = Image.open(image_path).convert("RGB")

        # Prepare input
        inputs = feature_extractor(images=image, return_tensors="tf")
        outputs = model(**inputs)
        logits = outputs.logits

        # Get the predicted class for each pixel
        predicted_mask = tf.argmax(logits, axis=1)[0].numpy()

        # Filter classes outside the ADE20K range (0-149)
        predicted_mask = np.where(predicted_mask > 149, 0, predicted_mask)

        return predicted_mask

    except Exception as e:
        logging.error(f"Error classifying image {image_path}: {e}")
        return None

def save_segmentation_in_band(predicted_mask, original_image, original_size, output_path, target_band=0):
    """
    Save segmentation result in one of the image bands
    
    Args:
        predicted_mask: Predicted mask (numpy array)
        original_image: Original image (PIL Image)
        original_size: Original image size (tuple)
        output_path: Save path
        target_band: Target band number (0=Red, 1=Green, 2=Blue)
    """
    try:
        # Resize mask to original image size
        mask_image = Image.fromarray(predicted_mask.astype(np.uint8))
        mask_resized = mask_image.resize(original_size, Image.NEAREST)
        mask_array = np.array(mask_resized)
        
        # Convert original image to numpy array
        original_array = np.array(original_image.convert('RGB'))
        
        # Copy original image for modification
        modified_image_array = original_array.copy()
        
        # Number of classes in ADE20K (0-149)
        num_classes = 150
        
        # Normalize class values to 0-255 range
        normalized_classes = (mask_array * (255 / (num_classes - 1))).astype(np.uint8)
        
        # Place classification values in specified band
        modified_image_array[:, :, target_band] = normalized_classes
        
        # Save modified image
        output_image = Image.fromarray(modified_image_array)
        output_image.save(output_path)
        
        logging.info(f"✅ Saved band-modified image to {output_path}")
        return output_path
        
    except Exception as e:
        logging.error(f"Error saving segmentation in band: {e}")
        return None

def create_colored_mask_and_blend(original_image_path, raw_mask, blended_output_folder):
    """
    Creates a colored segmentation mask and blends it with the original image.
    Only used when blended images are requested.
    """
    try:
        original_image = Image.open(original_image_path).convert("RGB")
        original_size = original_image.size

        # Resize raw mask to original image size before coloring/blending
        mask_image = Image.fromarray(raw_mask.astype(np.uint16), mode='I;16')
        mask_image = mask_image.resize(original_size, resample=Image.NEAREST)
        resized_raw_mask = np.array(mask_image)

        # Create a colored visualization of the mask
        norm_mask = resized_raw_mask.astype(np.float32) / 149.0
        colormap = plt.get_cmap('gist_ncar', 150)
        colored_array = colormap(norm_mask)[:, :, :3]
        colored_array = (colored_array * 255).astype(np.uint8)
        color_image = Image.fromarray(colored_array, mode='RGB')

        # Blend the original image and the colored mask with 50% opacity
        blended = Image.blend(original_image, color_image, alpha=0.5)

        # Define the save path for the blended image
        base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
        blended_save_path = os.path.join(blended_output_folder, f"{base_filename}_blended.png")

        # Ensure the output folder exists
        os.makedirs(blended_output_folder, exist_ok=True)

        # Save the blended image
        blended.save(blended_save_path)
        return blended_save_path

    except Exception as e:
        logging.error(f"Error creating colored mask and blend for {original_image_path}: {e}")
        return None

def save_band_modified_image(original_image_path, raw_mask, band_output_folder, target_band=0):
    """
    Helper function to save image with modified band
    """
    try:
        original_image = Image.open(original_image_path).convert("RGB")
        original_size = original_image.size
        
        # Define save path
        base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
        band_save_path = os.path.join(band_output_folder, f"{base_filename}_band_{target_band}.png")
        
        # Ensure output folder exists
        os.makedirs(band_output_folder, exist_ok=True)
        
        # Use save_segmentation_in_band function
        return save_segmentation_in_band(raw_mask, original_image, original_size, band_save_path, target_band)
        
    except Exception as e:
        logging.error(f"Error saving band modified image: {e}")
        return None

def classify_images_in_folder_with_band(image_folder, band_output_folder, target_band=0, 
                                      create_blended=False, blended_output_folder=None):
    """
    Classify images in folder with band modification capability
    
    Args:
        image_folder: Input images folder
        band_output_folder: Output folder for band-modified images
        target_band: Target band number (0=Red, 1=Green, 2=Blue)
        create_blended: Whether to create blended images (default: False)
        blended_output_folder: Output folder for blended images (required if create_blended=True)
    
    Returns:
        dict: Dictionary containing paths of processed images
    """
    logging.info(f"Starting image classification with band modification for images in: {image_folder}")
    os.makedirs(band_output_folder, exist_ok=True)
    
    if create_blended:
        if not blended_output_folder:
            raise ValueError("blended_output_folder must be provided when create_blended=True")
        os.makedirs(blended_output_folder, exist_ok=True)

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in supported_extensions]

    if not image_files:
        logging.warning(f"No supported image files found in {image_folder}.")
        return {"blended_paths": [], "band_paths": []}

    # Load model and feature extractor once
    try:
        load_model_and_feature_extractor()
    except Exception:
        logging.error("Failed to load classification model.")
        return {"blended_paths": [], "band_paths": []}

    logging.info(f"Found {len(image_files)} images to process.")
    
    results = {
        "blended_paths": [],
        "band_paths": []
    }

    for image_filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(image_folder, image_filename)

        # Classify image
        predicted_mask = classify_image(image_path)

        if predicted_mask is not None:
            # Save image with modified band
            band_path = save_band_modified_image(image_path, predicted_mask, band_output_folder, target_band)
            if band_path:
                results["band_paths"].append(band_path)
            
            # Create blended image only if requested
            if create_blended and blended_output_folder:
                blended_path = create_colored_mask_and_blend(image_path, predicted_mask, blended_output_folder)
                if blended_path:
                    results["blended_paths"].append(blended_path)

    return results

def classify_images_in_folder(image_folder, blended_output_folder):
    """
    Original function for backward compatibility - creates blended images only
    """
    result = classify_images_in_folder_with_band(
        image_folder, 
        band_output_folder=None,  # Not creating band images in original function
        create_blended=True, 
        blended_output_folder=blended_output_folder
    )
    return result["blended_paths"]

def analyze_band_modification(original_image_path, modified_image_path, target_band=0):
    """
    Analyze modified image to check class distribution in band
    """
    if not os.path.exists(modified_image_path):
        print(f"Modified file not found: {modified_image_path}")
        return
    
    # Load images
    original_img = Image.open(original_image_path).convert('RGB')
    modified_img = Image.open(modified_image_path).convert('RGB')
    
    original_array = np.array(original_img)
    modified_array = np.array(modified_img)
    
    # Extract target band
    original_band = original_array[:, :, target_band]
    modified_band = modified_array[:, :, target_band]
    
    # Show class statistics
    unique_values, counts = np.unique(modified_band, return_counts=True)
    
    band_names = ['Red', 'Green', 'Blue']
    print(f"\n=== {band_names[target_band]} Band Analysis ===")
    print(f"File: {os.path.basename(modified_image_path)}")
    print(f"Number of classes found: {len(unique_values)}")
    print(f"Class statistics in band:")
    
    for value, count in zip(unique_values, counts):
        # Convert band value to approximate class number
        class_num = int(value * 149 / 255)
        percentage = (count / (modified_band.shape[0] * modified_band.shape[1])) * 100
        print(f"  Class {class_num:2d} | Band value: {value:3d} | Pixels: {count:8d} | Percent: {percentage:5.2f}%")

# Example usage functions
def process_images_band_only(image_folder="test_images", output_folder="band_output", target_band=0):
    """
    Process images with band modification only (no blended images)
    """
    if not os.path.exists(image_folder):
        print(f"Image folder '{image_folder}' not found.")
        return
    
    print(f"Processing images in: {image_folder}")
    print(f"Target band: {['Red', 'Green', 'Blue'][target_band]}")
    
    results = classify_images_in_folder_with_band(
        image_folder=image_folder,
        band_output_folder=output_folder,
        target_band=target_band,
        create_blended=False  # No blended images
    )
    
    print(f"Processed {len(results['band_paths'])} images with band modification.")
    return results["band_paths"]

def process_images_with_options(image_folder="test_images", 
                               band_output_folder="band_output", 
                               target_band=0,
                               create_blended=False,
                               blended_output_folder="blended_output"):
    """
    Process images with full options
    """
    if not os.path.exists(image_folder):
        print(f"Image folder '{image_folder}' not found.")
        return
    
    print(f"Processing images in: {image_folder}")
    print(f"Target band: {['Red', 'Green', 'Blue'][target_band]}")
    print(f"Create blended images: {create_blended}")
    
    results = classify_images_in_folder_with_band(
        image_folder=image_folder,
        band_output_folder=band_output_folder,
        target_band=target_band,
        create_blended=create_blended,
        blended_output_folder=blended_output_folder if create_blended else None
    )
    
    print(f"Processed {len(results['band_paths'])} images with band modification.")
    if create_blended:
        print(f"Created {len(results['blended_paths'])} blended images.")
    
    return results

if __name__ == "__main__":
    # Example 1: Band modification only (no blended images)
    print("=== Example 1: Band modification only ===")
    process_images_band_only(
        image_folder="test_images",
        output_folder="band_output_red",
        target_band=0  # Red band
    )
    
    # Example 2: Band modification + blended images (optional)
    print("\n=== Example 2: With options ===")
    process_images_with_options(
        image_folder="test_images",
        band_output_folder="band_output_blue",
        target_band=2,  # Blue band
        create_blended=True,  # Enable blended images
        blended_output_folder="blended_output"
    )
