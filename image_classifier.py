import os
from PIL import Image, ExifTags
import numpy as np
import tensorflow as tf
from transformers import (
    TFSegformerForSemanticSegmentation,
    SegformerImageProcessor
)
import logging
import json
from tqdm import tqdm
import matplotlib.pyplot as plt  # Import matplotlib for colormap
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import ndimage
import threading

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
MODELS_BASE_DIR = os.path.join(BASE_DIR, "saved_model")
DEFAULT_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

os.makedirs(MODELS_BASE_DIR, exist_ok=True)

import re


def get_model_dir(model_name: str) -> str:
    match = re.search(r"segformer-(b\d)", model_name.lower())
    sub = match.group(1) if match else model_name.replace("/", "_")
    return os.path.join(MODELS_BASE_DIR, sub)


class SegformerClassifier:
    """Thread-safe Segformer-based image classifier."""

    def __init__(self, default_model_name: str = DEFAULT_MODEL_NAME):
        self.model = None
        self.feature_extractor = None
        self.loaded_model_name = None
        self.default_model_name = default_model_name
        self._lock = threading.Lock()

    def load_model_and_feature_extractor(self, model_name: str | None = None):
        """Load the Segformer model and feature extractor in a thread-safe manner."""
        model_name = model_name or self.default_model_name
        with self._lock:
            if self.loaded_model_name != model_name:
                self.model = None
                self.feature_extractor = None
                self.loaded_model_name = model_name

            model_dir = get_model_dir(model_name)
            os.makedirs(model_dir, exist_ok=True)

            logging.info("📦 Loading Segformer model and feature extractor...")

            try:
                tf_model_path = os.path.join(model_dir, "tf_model.h5")
                if os.path.exists(tf_model_path):
                    logging.info("📦 Found model locally. Loading from disk...")
                    self.model = TFSegformerForSemanticSegmentation.from_pretrained(model_dir)
                else:
                    logging.info("⬇ Model not found locally. Downloading from Hugging Face...")
                    self.model = TFSegformerForSemanticSegmentation.from_pretrained(model_name, from_pt=True)
                    self.model.save_pretrained(model_dir)
                    logging.info("✅ Model downloaded and saved locally.")

                preprocessor_path = os.path.join(model_dir, "preprocessor_config.json")
                if os.path.exists(preprocessor_path):
                    logging.info("📦 Found feature extractor locally.")
                    self.feature_extractor = SegformerImageProcessor.from_pretrained(model_dir)
                else:
                    logging.warning("⚠️ Feature extractor not found locally. Downloading from Hugging Face...")
                    self.feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
                    self.feature_extractor.save_pretrained(model_dir)
                    logging.info("✅ Feature extractor downloaded and saved locally.")

            except Exception as e:
                logging.error(f"❌ Error loading model or feature extractor: {e}")
                self.model = None
                self.feature_extractor = None
                raise

    def classify_image(self, image_path: str, model_name: str | None = None):
        """Classify a single image and return the predicted segmentation mask."""
        model_name = model_name or self.default_model_name
        if (
            self.model is None
            or self.feature_extractor is None
            or self.loaded_model_name != model_name
        ):
            self.load_model_and_feature_extractor(model_name)

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.feature_extractor(images=image, return_tensors="tf")
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_mask = tf.argmax(logits, axis=1)[0].numpy()
            predicted_mask = np.where(predicted_mask > 149, 0, predicted_mask)
            return predicted_mask
        except Exception as e:
            logging.error(f"Error classifying image {image_path}: {e}")
            return None

    def process_single_image(self, image_path, blended_output_folder, model_name: str | None = None):
        """Classify a single image and return its blended path with EXIF data."""
        predicted_mask = self.classify_image(image_path, model_name)
        exif = extract_exif_data(image_path)
        blended_path = None
        if predicted_mask is not None:
            blended_path = create_colored_mask_and_blend(
                image_path, predicted_mask, blended_output_folder
            )
        return {"original_path": image_path, "blended_path": blended_path, "exif": exif}

    def classify_images_in_folder(self, image_folder, blended_output_folder, model_name: str | None = None):
        """Classify all images in a folder and generate blended outputs."""
        model_name = model_name or self.default_model_name
        logging.info(f"Starting image classification and blending for images in: {image_folder}")
        os.makedirs(blended_output_folder, exist_ok=True)

        supported_extensions = {".jpg", ".jpeg", ".png"}
        image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in supported_extensions]
        if not image_files:
            logging.warning(f"No supported image files found in {image_folder}.")
            return []

        try:
            self.load_model_and_feature_extractor(model_name)
        except Exception:
            logging.error("Failed to load classification model. Skipping classification and blending.")
            return []

        logging.info(f"Found {len(image_files)} images to classify and blend.")
        results = []
        image_paths = [os.path.join(image_folder, f) for f in image_files]
        num_workers = max(1, (os.cpu_count() or 1) - 1)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.process_single_image, path, blended_output_folder, model_name): path
                for path in image_paths
            }
            with tqdm(total=len(futures), desc="Classifying and Blending Images") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result.get("blended_path"):
                        results.append(result)
                    pbar.update(1)

        results_json_path = os.path.join(blended_output_folder, "classification_results.json")
        try:
            with open(results_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save classification results JSON: {e}")

        return results


def apply_neighborhood_smoothing(mask: np.ndarray, kernel_size: int = 3, num_classes: int = 150) -> np.ndarray:
    """Apply a simple neighbourhood mode filter to reduce isolated misclassifications.

    Each pixel is replaced by the most common class within a square window around it.
    This helps merge small outlier regions with the surrounding background.

    Args:
        mask: 2D array of class indices.
        kernel_size: Size of the square neighbourhood; must be odd.
        num_classes: Total number of classes in the segmentation mask.

    Returns:
        np.ndarray: Smoothed mask with reduced outlier pixels.
    """

    # One-hot encode the mask so we can count occurrences of each class locally
    mask_tensor = tf.one_hot(mask, depth=num_classes, dtype=tf.float32)
    mask_tensor = tf.expand_dims(mask_tensor, axis=0)  # Add batch dimension

    # Depthwise convolution with an all-ones kernel counts class occurrences in the neighbourhood
    kernel = tf.ones((kernel_size, kernel_size, num_classes, 1), dtype=tf.float32)
    counts = tf.nn.depthwise_conv2d(mask_tensor, kernel, strides=[1, 1, 1, 1], padding="SAME")

    # Argmax across classes gives the most frequent class in the neighbourhood
    counts = tf.reshape(counts, (mask.shape[0], mask.shape[1], num_classes))
    smoothed = tf.argmax(counts, axis=-1)

    return smoothed.numpy().astype(mask.dtype)


def fill_holes_in_class(
    mask: np.ndarray,
    target_classes,
    structure_size: int = 3,
    area_threshold: int = 0,
) -> np.ndarray:
    """Fill internal holes within specified classes of a segmentation mask.

    Args:
        mask: 2D array of class indices.
        target_classes: Iterable of class IDs for which holes should be filled.
        structure_size: Size of the structuring element used for hole filling.
        area_threshold: Maximum area (in pixels) of holes to fill. If 0, all holes
            are filled regardless of size.

    Returns:
        np.ndarray: Mask with holes in the target classes filled.
    """

    # Copy to avoid modifying the input array in-place
    filled_mask = mask.copy()
    structure = np.ones((structure_size, structure_size), dtype=bool)

    for cls in target_classes:
        class_region = filled_mask == cls
        # Fill holes inside the class region
        filled_region = ndimage.binary_fill_holes(class_region, structure=structure)
        holes = filled_region & (~class_region)

        if area_threshold > 0:
            # Keep only holes smaller than the area threshold
            labeled, num_features = ndimage.label(holes)
            small_holes = np.zeros_like(holes, dtype=bool)
            for i in range(1, num_features + 1):
                region = labeled == i
                if np.sum(region) <= area_threshold:
                    small_holes |= region
            holes = small_holes

        # Assign the surrounding class to the hole pixels
        filled_mask[holes] = cls

    return filled_mask


def extract_exif_data(image_path: str) -> dict:
    """Extract EXIF metadata from an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary mapping EXIF tags to their values. Returns an empty
        dict if no EXIF data is found or an error occurs.
    """
    try:
        img = Image.open(image_path)
        exif_data = {}
        info = img._getexif()
        if info:
            for tag_id, value in info.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[tag] = str(value)
        return exif_data
    except Exception as e:
        logging.warning(f"Could not extract EXIF data from {image_path}: {e}")
        return {}

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
        original_image = Image.open(original_image_path)
        original_exif = original_image.info.get("exif")
        original_image = original_image.convert("RGB")
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
            if original_exif:
                blended.save(blended_save_path, exif=original_exif)
            else:
                blended.save(blended_save_path)
            # logging.info(f"✅ Saved blended image for {base_filename} to {blended_save_path}")
            return blended_save_path # Return the path to the saved blended image
        except Exception as e:
            logging.error(f"Error saving blended image for {base_filename}: {e}")
            return None

     except Exception as e:
        logging.error(f"Error creating green channel overlay for {original_image_path}: {e}")
        return None
if __name__ == "__main__":
    # Example usage: classify images in 'test_images', save blended to 'test_blended_masks'
    test_image_folder = "test_images"
    test_blended_output_folder = "test_blended_masks"
    if os.path.exists(test_image_folder):
        print(f"Running example classification for folder: {test_image_folder}")
        classifier = SegformerClassifier()
        blended_paths = classifier.classify_images_in_folder(
            test_image_folder, test_blended_output_folder
        )
        print(
            f"Generated {len(blended_paths)} blended images in {test_blended_output_folder}"
        )
    else:
        print(f"Test image folder '{test_image_folder}' not found. Skipping example.")
