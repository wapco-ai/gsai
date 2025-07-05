import os
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import TFSegformerForSemanticSegmentation, SegformerFeatureExtractor
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt  # Import matplotlib for colormap

# Configure TensorFlow to utilise GPU when available
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logging.info(f"Using GPU devices: {[gpu.name for gpu in gpus]}")
#     except Exception as e:
#         logging.warning(f"Could not set GPU memory growth: {e}")
# else:
#     logging.info("No GPU detected. Using CPU only.")

# Configure logging for the classifier
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model and feature extractor paths
# Assuming the model will be saved in a 'saved_model' directory relative to this script
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
                # Specify from_pt=True to load PyTorch weights
                # Specify a cache_dir if you want to control where models are downloaded
                model = TFSegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, from_pt=True)
                model.save_pretrained(MODEL_DIR)
                logging.info("✅ Model downloaded and saved.")
            else:
                logging.info("📦 Loading model from local path...")
                model = TFSegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
                logging.info("✅ Model loaded from local path.")

            # Try loading the feature extractor from the same local directory as the model
            if os.path.exists(MODEL_DIR):
                try:
                    feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_DIR)
                    logging.info("✅ Feature extractor loaded from local path.")
                except Exception as e:
                    logging.warning(
                        f"Could not load feature extractor from local path: {e}. Falling back to Hugging Face hub."
                    )
                    feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
                    logging.info("✅ Feature extractor downloaded and loaded.")
            else:
                feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
                logging.info("✅ Feature extractor downloaded and loaded.")

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
        # The feature extractor handles resizing and normalization
        inputs = feature_extractor(images=image, return_tensors="tf")
        outputs = model(**inputs)
        logits = outputs.logits

        # Get the predicted class for each pixel
        # Resize logits to original image size before argmax is more accurate,
        # but sticking to sample logic for now (argmax then resize mask)
        # If high accuracy is needed, resize logits using tf.image.resize
        # For simplicity, we will get the raw mask and resize the *output* mask
        predicted_mask = tf.argmax(logits, axis=1)[0].numpy()

        # Filter classes outside the ADE20K range (0-149) as per sample
        # Replace Unknowns with background (class 0)
        predicted_mask = np.where(predicted_mask > 149, 0, predicted_mask)

        return predicted_mask

    except Exception as e:
        logging.error(f"Error classifying image {image_path}: {e}")
        return None

# --- New function to generate visualizations and blended image ---
def generate_visualizations_and_blend(original_image_path, raw_mask, output_mask_folder):
    """
    Generates a colored segmentation mask and a blended image.

    Args:
        original_image_path (str): Path to the original image.
        raw_mask (np.ndarray): The raw segmentation mask array.
        output_mask_folder (str): Folder to save the generated images.

    Returns:
        str or None: Path to the saved blended image, or None if an error occurred.
    """
    try:
        original_image = Image.open(original_image_path).convert("RGB")
        original_size = original_image.size # (width, height)

        # Ensure mask is resized to original image size before coloring for better visualization
        # Use NEAREST neighbor for resizing masks to maintain class IDs
        mask_image = Image.fromarray(raw_mask.astype(np.uint16), mode='I;16')
        mask_image = mask_image.resize(original_size, resample=Image.NEAREST)
        resized_raw_mask = np.array(mask_image) # Get the resized mask as numpy array


        # Create a colored visualization of the mask
        # Use a colormap with 150 distinct colors for ADE20K classes (0-149)
        # Ensure the mask values are normalized to the range [0, 1] for the colormap
        norm_mask = resized_raw_mask.astype(np.float32) / 149.0 # Normalize to [0, 1]
        colormap = plt.get_cmap('gist_ncar', 150)
        colored_array = colormap(norm_mask)[:, :, :3]  # Get RGB, discard alpha
        colored_array = (colored_array * 255).astype(np.uint8) # Scale to 0-255 and convert to uint8
        color_image = Image.fromarray(colored_array, mode='RGB')


        # Define output paths
        base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
        colored_mask_path = os.path.join(output_mask_folder, f"{base_filename}_colored_mask.png")
        blended_path = os.path.join(output_mask_folder, f"{base_filename}_blended.png")
        raw_mask_save_path = os.path.join(output_mask_folder, f"{base_filename}_mask.png") # Also save the raw mask


        # Save the colored mask
        try:
             color_image.save(colored_mask_path)
             # logging.info(f"Saved colored mask to {colored_mask_path}")
        except Exception as e:
             logging.error(f"Error saving colored mask for {base_filename}: {e}")


        # Save the raw mask (as 16-bit PNG to preserve class IDs)
        try:
            mask_image.save(raw_mask_save_path)
            # logging.info(f"Saved raw mask to {raw_mask_save_path}")
        except Exception as e:
            logging.error(f"Error saving raw mask for {base_filename}: {e}")


        # Blend the original image and the colored mask with 50% opacity
        blended = Image.blend(original_image, color_image, alpha=0.5)

        # Save the blended image
        try:
            blended.save(blended_path)
            logging.info(f"✅ Saved blended image for {base_filename} to {blended_path}")
            return blended_path # Return the path to the blended image
        except Exception as e:
            logging.error(f"Error saving blended image for {base_filename}: {e}")
            return None

    except Exception as e:
        logging.error(f"Error generating visualizations and blend for {original_image_path}: {e}")
        return None


# --- Modify classify_images_in_folder to use the new function ---
def classify_images_in_folder(image_folder, output_mask_folder, blended_output_folder):
    """
    Classifies all images in a folder, saves segmentation masks, colored masks,
    and blended images, and returns paths to the blended images.

    Args:
        image_folder (str): Path to the folder containing input images.
        output_mask_folder (str): Path to the folder where masks will be saved.
        blended_output_folder (str): Path to the folder where blended images will be saved.

    Returns:
        list[str]: A list of paths to the saved blended images.
    """
    logging.info(f"Starting image classification for images in: {image_folder}")
    os.makedirs(output_mask_folder, exist_ok=True)
    os.makedirs(blended_output_folder, exist_ok=True) # Ensure blended output folder exists

    supported_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in supported_extensions]

    if not image_files:
        logging.warning(f"No supported image files found in {image_folder}.")
        return [] # Return empty list if no images

    # Load model and feature extractor once before processing all images
    try:
        load_model_and_feature_extractor()
    except Exception:
        logging.error("Failed to load classification model. Skipping classification and blending.")
        return [] # Return empty list on model load failure

    logging.info(f"Found {len(image_files)} images to classify.")
    blended_image_paths = []

    for image_filename in tqdm(image_files, desc="Classifying and Blending Images"):
        image_path = os.path.join(image_folder, image_filename)

        predicted_mask = classify_image(image_path)

        if predicted_mask is not None:
            # Generate visualizations and the blended image
            blended_path = generate_visualizations_and_blend(image_path, predicted_mask, output_mask_folder)
            if blended_path:
                # We need to copy or move the blended image to the dedicated blended_output_folder
                # For simplicity, let's just save it directly to the blended_output_folder
                # Let's update generate_visualizations_and_blend to save to a specific folder
                # instead of the mask output folder.

                # Let's restructure: generate_visualizations_and_blend will save to a given folder
                # and classify_images_in_folder will pass the correct folder.

                # Re-call generate_visualizations_and_blend, this time saving to the dedicated folder
                # We could also modify generate_visualizations_and_blend to take two output folders
                # Or modify it to return the PIL Image and save here.
                # Let's keep it simple and modify generate_visualizations_and_blend
                # to save to `output_mask_folder` for masks and `blended_output_folder` for blended.
                # This requires passing both folders.

                # --- Revised logic: Pass both folders to a new function ---
                # Let's create a new function `process_image_with_segmentation`
                # that encapsulates classify, visualize, and blend.

                # For now, let's stick to the plan and modify generate_visualizations_and_blend
                # to save the blended image to blended_output_folder directly.
                # This will require passing blended_output_folder to it.

                # Modify generate_visualizations_and_blend to take blended_output_folder
                # and update the saving path for blended_path.

                # --- Let's restart the modification plan for image_classifier.py ---
                # Okay, the plan to modify generate_visualizations_and_blend to take two folders
                # or return PIL images makes the code more complex.
                # Let's simplify:
                # 1. classify_image returns the raw mask.
                # 2. A new function `process_and_visualize` takes original image path, raw mask,
                #    mask output folder, and blended output folder. It saves all outputs.
                # 3. classify_images_in_folder calls classify_image and then process_and_visualize,
                #    and collects the paths of the blended images.

                # Let's refactor image_classifier.py with this simpler structure.
                # The code block above already started this, but the saving part needs adjustment.

                # Let's refine the `generate_visualizations_and_blend` function name and arguments.
                # How about `save_segmentation_outputs`? It takes original image path, raw mask, and
                # the base output directory for this image. It then saves mask, colored mask, and blended.
                # We need the path to the original image to get size and filename.

                # --- Re-revising image_classifier.py structure ---

                # `classify_image(image_path)`: Returns raw mask array. (Already done)
                # `process_image_outputs(original_image_path, raw_mask, output_base_dir)`:
                #   Takes original image path, raw mask, and a directory where all outputs for *this specific image* will go.
                #   Inside: creates raw mask, colored mask, blended image paths within `output_base_dir`.
                #   Saves all three.
                #   Returns path to the blended image.
                # `classify_images_in_folder(image_folder, output_base_folder_for_all_images)`:
                #   Iterates through image_folder.
                #   For each image, calls `classify_image`.
                #   Creates a subdirectory for the current image within `output_base_folder_for_all_images`.
                #   Calls `process_image_outputs` for the current image, passing the raw mask and the image-specific output subdirectory.
                #   Collects the blended image paths.
                #   Returns the list of blended image paths.

                # This structure seems cleaner. The Metashape script will then be pointed to the `output_base_folder_for_all_images`,
                # and it will need to find the blended images within the subdirectories.
                # OR, Metashape takes a list of image paths. The `classify_images_in_folder` can return the list of blended image paths directly.
                # Let's go with returning the list of blended paths.

                # --- Final Plan for image_classifier.py ---
                # 1. Keep `load_model_and_feature_extractor`.
                # 2. Keep `classify_image(image_path)` returning raw mask array.
                # 3. Create `create_colored_mask_and_blend(original_image_path, raw_mask, blended_save_path)`:
                #    Takes original image path, raw mask, and the *exact path* where the blended image should be saved.
                #    Generates colored mask internally (can save it alongside blended if desired, but primarily for blending).
                #    Blends and saves the blended image to `blended_save_path`.
                #    Does NOT save the raw mask or colored mask to a separate mask folder in this function.
                #    Returns `blended_save_path`.
                # 4. Modify `classify_images_in_folder(image_folder, blended_output_folder)`:
                #    Takes input `image_folder` and the dedicated `blended_output_folder`.
                #    Creates `blended_output_folder` if it doesn't exist.
                #    Iterates through images in `image_folder`.
                #    For each image:
                #        Calls `classify_image` to get the raw mask.
                #        Constructs the desired `blended_save_path` within `blended_output_folder`.
                #        Calls `create_colored_mask_and_blend`.
                #        Collects the `blended_save_path` in a list.
                #    Returns the list of `blended_save_path`s.

                # This is cleaner. The raw masks can be saved as a separate step if needed, but the user only asked for colored and blended.
                # Let's implement `create_colored_mask_and_blend`.

                blended_path = create_colored_mask_and_blend(image_path, predicted_mask, blended_output_folder)
                if blended_path:
                    blended_image_paths.append(blended_path)
            # No else needed, error is logged in generate_visualizations_and_blend

    return blended_image_paths # Return the list of blended image paths

# --- New function to create colored mask and blend ---
def create_colored_mask_and_blend(original_image_path, raw_mask, blended_output_folder):
     """
    Creates an output image where the original RGB image is preserved except
    for the green channel, which is replaced with the predicted class ID for
    each pixel. The modified image is saved to the specified output folder.

     Args:
         original_image_path (str): Path to the original image.
         raw_mask (np.ndarray): The raw segmentation mask array.
         blended_output_folder (str): Path to the folder where the blended image will be saved.

     Returns:
         str or None: Path to the saved blended image, or None if an error occurred.
     """
     try:
        original_image = Image.open(original_image_path).convert("RGB")
        original_size = original_image.size # (width, height)

        # Resize raw mask to match the original image size
        mask_image = Image.fromarray(raw_mask.astype(np.uint16), mode='I;16')
        mask_image = mask_image.resize(original_size, resample=Image.NEAREST)
        resized_raw_mask = np.array(mask_image, dtype=np.uint8)

        # Insert the mask values into the green channel of the original image
        original_array = np.array(original_image)
        original_array[:, :, 1] = resized_raw_mask
        blended = Image.fromarray(original_array, mode="RGB")

        # Define the save path for the blended image within the designated blended output folder
        base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
        blended_save_path = os.path.join(blended_output_folder, f"{base_filename}_blended.png")

        # Ensure the output folder exists
        os.makedirs(blended_output_folder, exist_ok=True)

        # Save the blended image
        try:
            blended.save(blended_save_path)
            # logging.info(f"✅ Saved blended image for {base_filename} to {blended_save_path}")
            return blended_save_path # Return the path to the saved blended image
        except Exception as e:
            logging.error(f"Error saving blended image for {base_filename}: {e}")
            return None

     except Exception as e:
        logging.error(f"Error creating green channel overlay for {original_image_path}: {e}")
        return None


# --- Modify classify_images_in_folder again to use create_colored_mask_and_blend ---
def classify_images_in_folder(image_folder, blended_output_folder):
    """
    Classifies all images in a folder, generates and saves blended images,
    and returns paths to the blended images.

    Args:
        image_folder (str): Path to the folder containing input images.
        blended_output_folder (str): Path to the folder where blended images will be saved.

    Returns:
        list[str]: A list of paths to the saved blended images.
    """
    logging.info(f"Starting image classification and blending for images in: {image_folder}")
    os.makedirs(blended_output_folder, exist_ok=True) # Ensure blended output folder exists

    supported_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in supported_extensions]

    if not image_files:
        logging.warning(f"No supported image files found in {image_folder}.")
        return [] # Return empty list if no images

    # Load model and feature extractor once before processing all images
    try:
        load_model_and_feature_extractor()
    except Exception:
        logging.error("Failed to load classification model. Skipping classification and blending.")
        return [] # Return empty list on model load failure

    logging.info(f"Found {len(image_files)} images to classify and blend.")
    blended_image_paths = []

    for image_filename in tqdm(image_files, desc="Classifying and Blending Images"):
        image_path = os.path.join(image_folder, image_filename)

        predicted_mask = classify_image(image_path)

        if predicted_mask is not None:
            # Create colored mask and blended image, save blended image
            blended_path = create_colored_mask_and_blend(image_path, predicted_mask, blended_output_folder)
            if blended_path:
                blended_image_paths.append(blended_path)
            # Error saving blended image is logged within create_colored_mask_and_blend

    return blended_image_paths # Return the list of blended image paths


if __name__ == "__main__":
    # Example usage: classify images in 'test_images', save blended to 'test_blended_masks'
    # You would need to create these folders and place some images in 'test_images'
    test_image_folder = "test_images"
    test_blended_output_folder = "test_blended_masks"
    if os.path.exists(test_image_folder):
        print(f"Running example classification for folder: {test_image_folder}")
        blended_paths = classify_images_in_folder(test_image_folder, test_blended_output_folder)
        print(f"Generated {len(blended_paths)} blended images in {test_blended_output_folder}")
    else:
        print(f"Test image folder '{test_image_folder}' not found. Skipping example.")
