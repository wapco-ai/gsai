import os
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import TFSegformerForSemanticSegmentation, SegformerFeatureExtractor
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import time
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

model = None
feature_extractor = None
device_name = None
gpu_available = False
batch_size = 8

def detect_optimal_batch_size():
    global batch_size
    if not gpu_available:
        batch_size = 1
        return batch_size
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                gpu_details = tf.config.experimental.get_memory_info(gpus[0])
                total_memory = gpu_details['total'] / (1024**3)
                if total_memory >= 8:
                    batch_size = 16
                elif total_memory >= 6:
                    batch_size = 12
                elif total_memory >= 4:
                    batch_size = 8
                else:
                    batch_size = 4
                logging.info(f"🎮 GPU Memory: {total_memory:.1f}GB, Optimal batch size: {batch_size}")
            except:
                batch_size = 6
                logging.info(f"🎮 Using conservative batch size: {batch_size}")
    except Exception as e:
        logging.warning(f"Failed to detect GPU memory: {e}")
        batch_size = 4
    return batch_size

def setup_gpu_optimization():
    global device_name, gpu_available
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            policy = tf.keras.mixed_precision.Policy('float32')
            tf.keras.mixed_precision.set_global_policy(policy)
            device_name = '/GPU:0'
            gpu_available = True
            logging.info(f"🚀 GPU TURBO MODE ACTIVATED!")
            logging.info(f"🎮 GPUs configured: {len(gpus)}")
            logging.info(f"⚡ Mixed precision: DISABLED")
            logging.info(f"💻 Device: {device_name}")
            detect_optimal_batch_size()
        else:
            tf.config.threading.set_intra_op_parallelism_threads(6)
            tf.config.threading.set_inter_op_parallelism_threads(3)
            device_name = '/CPU:0'
            gpu_available = False
            batch_size = 2
            logging.info("💻 No GPU found - CPU optimized mode")
            logging.info(f"💻 Device: {device_name}")
    except Exception as e:
        logging.error(f"GPU setup failed: {e}")
        device_name = '/CPU:0'
        gpu_available = False
        batch_size = 1

def load_model_and_feature_extractor():
    global model, feature_extractor
    if model is None or feature_extractor is None:
        logging.info("🚀 Loading model with TURBO optimizations...")
        setup_gpu_optimization()
        try:
            logging.info("📦 Loading model from local directory...")
            with tf.device(device_name):
                model = TFSegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
            logging.info("✅ Model loaded from local directory.")

            # Load feature extractor manually from local JSON
            preprocessor_path = os.path.join(MODEL_DIR, "preprocessor_config.json")
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                feature_extractor = SegformerFeatureExtractor(**config_dict)
                logging.info("✅ Feature extractor loaded from local JSON config.")
            else:
                raise FileNotFoundError("preprocessor_config.json not found in saved_model directory.")

            logging.info(f"🔥 TURBO MODE READY - Batch size: {batch_size}" if gpu_available else f"💻 CPU MODE READY - Batch size: {batch_size}")
        except Exception as e:
            logging.error(f"Error loading model or feature extractor: {e}")
            model = None
            feature_extractor = None
            raise


def classify_image(image_path):
    if model is None or feature_extractor is None:
        load_model_and_feature_extractor()
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="tf")
        with tf.device(device_name):
            outputs = model(**inputs)
            logits = outputs.logits
        predicted_mask = tf.argmax(logits, axis=1)[0].numpy()
        predicted_mask = np.where(predicted_mask > 149, 0, predicted_mask)
        del inputs, outputs, logits
        gc.collect()
        return predicted_mask
    except Exception as e:
        logging.error(f"Error classifying image {image_path}: {e}")
        return None

def classify_images_in_folder(image_folder):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    all_results = {}
    load_model_and_feature_extractor()
    for img_path in tqdm(image_paths, desc="Classifying"):
        mask = classify_image(img_path)
        if mask is not None:
            all_results[img_path] = mask
    return all_results

def save_segmentation_to_band(image_path, mask, output_folder, target_band=0):
    try:
        os.makedirs(output_folder, exist_ok=True)
        original = Image.open(image_path).convert("RGB")
        original_array = np.array(original)
        resized_mask = Image.fromarray(mask.astype(np.uint8)).resize(original.size, Image.NEAREST)
        normalized = (np.array(resized_mask) * (255 / 149)).astype(np.uint8)
        modified = original_array.copy()
        modified[:, :, target_band] = normalized
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        Image.fromarray(modified).save(output_path)
        return output_path
    except Exception as e:
        logging.error(f"Error saving band image: {e}")
        return None

def create_colored_mask_and_blend(image_path, mask, output_folder):
    try:
        os.makedirs(output_folder, exist_ok=True)
        original = Image.open(image_path).convert("RGB")
        resized_mask = Image.fromarray(mask.astype(np.uint8)).resize(original.size, Image.NEAREST)
        norm_mask = np.array(resized_mask).astype(np.float32) / 149.0
        colormap = plt.get_cmap('gist_ncar', 150)
        colored_array = (colormap(norm_mask)[:, :, :3] * 255).astype(np.uint8)
        blended = Image.blend(original, Image.fromarray(colored_array), alpha=0.5)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        blended.save(output_path)
        return output_path
    except Exception as e:
        logging.error(f"Error creating blended image: {e}")
        return None

def visualize_mask(image_path, mask):
    try:
        original_image = Image.open(image_path).convert("RGB")
        original_size = original_image.size
        mask_resized = Image.fromarray(mask.astype(np.uint8)).resize(original_size, Image.NEAREST)
        norm_mask = np.array(mask_resized).astype(np.float32) / 149.0
        colormap = plt.get_cmap('gist_ncar', 150)
        colored_array = (colormap(norm_mask)[:, :, :3] * 255).astype(np.uint8)
        blended = Image.blend(original_image, Image.fromarray(colored_array), alpha=0.5)
        blended.show()
    except Exception as e:
        logging.error(f"Error visualizing mask: {e}")

if __name__ == "__main__":
    print("=== TURBO PERFORMANCE INFO ===")
    setup_gpu_optimization()
    print(f"Device: {device_name}, GPU Available: {gpu_available}, Batch Size: {batch_size}")

    test_folder = os.path.join(BASE_DIR, "test_images")
    output_folder = os.path.join(BASE_DIR, "output")

    if os.path.exists(test_folder):
        results = classify_images_in_folder(test_folder)
        for path, mask in results.items():
            save_segmentation_to_band(path, mask, os.path.join(output_folder, "bands"), target_band=0)
            create_colored_mask_and_blend(path, mask, os.path.join(output_folder, "blended"))
    else:
        logging.warning("Test image folder not found.")
