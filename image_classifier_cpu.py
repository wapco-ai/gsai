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

# Use the application's logging configuration

# Model and feature extractor paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# Global variables
model = None
feature_extractor = None
device_name = None
gpu_available = False
batch_size = 1  # Will be auto-adjusted based on GPU memory
# Allow overriding batch size via environment variable
_env_batch = os.environ.get("BATCH_SIZE")
if _env_batch:
    try:
        batch_size = int(_env_batch)
        logging.info(f"🔧 Using batch size from environment: {batch_size}")
    except ValueError:
        logging.warning(
            f"Invalid BATCH_SIZE value '{_env_batch}', falling back to auto detection"
        )

def detect_optimal_batch_size():
    """Detect optimal batch size based on GPU memory"""
    global batch_size

    # Respect manual override if provided
    if _env_batch:
        return batch_size
    
    if not gpu_available:
        batch_size = 1
        return batch_size
    
    try:
        # Get GPU memory info
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Try to get memory info (if available)
            try:
                gpu_details = tf.config.experimental.get_memory_info(gpus[0])
                total_memory = gpu_details['total'] / (1024**3)  # Convert to GB
                
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
                # Default conservative batch size
                batch_size = 6
                logging.info(f"🎮 Using conservative batch size: {batch_size}")
        
    except Exception as e:
        logging.warning(f"Failed to detect GPU memory: {e}")
        batch_size = 4
    
    return batch_size

def setup_gpu_optimization():
    """Setup aggressive GPU optimizations"""
    global device_name, gpu_available
    
    try:
        # Enable GPU memory growth and mixed precision
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            # Configure all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # Enable mixed precision for maximum speed
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            # Set device
            device_name = '/GPU:0'
            gpu_available = True
            
            logging.info(f"🚀 GPU TURBO MODE ACTIVATED!")
            logging.info(f"🎮 GPUs configured: {len(gpus)}")
            logging.info(f"⚡ Mixed precision: ENABLED")
            logging.info(f"💻 Device: {device_name}")
            
            # Detect optimal batch size
            detect_optimal_batch_size()
            
        else:
            # CPU optimization
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
    """Load model with aggressive GPU optimization"""
    global model, feature_extractor
    
    if model is None or feature_extractor is None:
        logging.info("🚀 Loading model with TURBO optimizations...")
        
        # Setup GPU first
        setup_gpu_optimization()
        
        try:
            # Load model strictly from local path
            if not os.path.exists(MODEL_DIR):
                raise FileNotFoundError(
                    f"Local model directory not found: {MODEL_DIR}. "
                    "Place the model in this directory to run without internet."
                )
            else:
                logging.info("📦 Loading model from local directory...")
                with tf.device(device_name):
                    model = TFSegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
                logging.info("✅ Model loaded from local directory.")

            # Load feature extractor from local JSON configuration
            preprocessor_path = os.path.join(MODEL_DIR, "preprocessor_config.json")
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                feature_extractor = SegformerFeatureExtractor(**config_dict)
                logging.info("✅ Feature extractor loaded from local JSON config.")
            else:
                raise FileNotFoundError(
                    "preprocessor_config.json not found in saved_model directory."
                )
            
            if gpu_available:
                logging.info(f"🔥 TURBO MODE READY - Batch size: {batch_size}")
            else:
                logging.info(f"💻 CPU MODE READY - Batch size: {batch_size}")

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            model = None
            feature_extractor = None
            raise

def classify_batch_ultra_fast(image_paths_batch):
    """Ultra-fast batch classification with GPU acceleration"""
    if model is None or feature_extractor is None:
        load_model_and_feature_extractor()
    
    try:
        # Load all images in batch
        images = []
        valid_paths = []
        
        for img_path in image_paths_batch:
            try:
                image = Image.open(img_path).convert("RGB")
                images.append(image)
                valid_paths.append(img_path)
            except Exception as e:
                logging.warning(f"Failed to load {img_path}: {e}")
                continue
        
        if not images:
            return {}
        
        # Prepare batch input - OPTIMIZED
        inputs = feature_extractor(images=images, return_tensors="tf")
        
        # GPU batch inference - MAXIMUM SPEED
        with tf.device(device_name):
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Process all masks in batch - VECTORIZED
        predicted_masks = tf.argmax(logits, axis=1).numpy()
        
        # Filter invalid classes - VECTORIZED
        predicted_masks = np.where(predicted_masks > 149, 0, predicted_masks)
        
        # Create results dictionary
        results = {}
        for i, img_path in enumerate(valid_paths):
            results[img_path] = predicted_masks[i]
        
        # Aggressive memory cleanup
        del inputs, outputs, logits, predicted_masks, images
        gc.collect()
        
        return results
        
    except Exception as e:
        logging.error(f"Batch classification failed: {e}")
        return {}

def save_results_batch_optimized(results, band_output_folder, target_band=0, 
                                create_blended=False, blended_output_folder=None):
    """Optimized batch result saving"""
    
    band_paths = []
    blended_paths = []
    
    # Ensure directories exist
    os.makedirs(band_output_folder, exist_ok=True)
    if create_blended and blended_output_folder:
        os.makedirs(blended_output_folder, exist_ok=True)
    
    for img_path, predicted_mask in results.items():
        try:
            # Save band-modified image
            band_path = save_band_modified_image_fast(img_path, predicted_mask, band_output_folder, target_band)
            if band_path:
                band_paths.append(band_path)
            
            # Save blended image if requested
            if create_blended and blended_output_folder:
                blended_path = create_colored_mask_and_blend_fast(img_path, predicted_mask, blended_output_folder)
                if blended_path:
                    blended_paths.append(blended_path)
                    
        except Exception as e:
            logging.error(f"Failed to save results for {img_path}: {e}")
    
    return {"band_paths": band_paths, "blended_paths": blended_paths}

def save_band_modified_image_fast(image_path, predicted_mask, band_output_folder, target_band=0):
    """Ultra-fast band modification saving"""
    try:
        # Load original image
        original_image = Image.open(image_path).convert("RGB")
        original_size = original_image.size
        original_array = np.array(original_image)
        
        # Resize mask - OPTIMIZED
        mask_resized = Image.fromarray(predicted_mask.astype(np.uint8)).resize(original_size, Image.NEAREST)
        mask_array = np.array(mask_resized)
        
        # Vectorized normalization - FASTEST
        normalized_classes = (mask_array * (255 / 149)).astype(np.uint8)
        
        # Copy and modify - OPTIMIZED
        modified_array = original_array.copy()
        modified_array[:, :, target_band] = normalized_classes
        
        # Save path
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(band_output_folder, f"{base_filename}_band_{target_band}.png")
        
        # Save with optimization
        Image.fromarray(modified_array).save(save_path, optimize=True, compress_level=1)
        
        return save_path
        
    except Exception as e:
        logging.error(f"Fast band save failed for {image_path}: {e}")
        return None

def create_colored_mask_and_blend_fast(image_path, raw_mask, blended_output_folder):
    """Ultra-fast blended image creation"""
    try:
        # Load original
        original_image = Image.open(image_path).convert("RGB")
        original_size = original_image.size
        
        # Resize mask
        mask_resized = Image.fromarray(raw_mask.astype(np.uint8)).resize(original_size, Image.NEAREST)
        resized_mask = np.array(mask_resized)
        
        # Vectorized coloring - OPTIMIZED
        norm_mask = resized_mask.astype(np.float32) / 149.0
        colormap = plt.get_cmap('gist_ncar', 150)
        colored_array = (colormap(norm_mask)[:, :, :3] * 255).astype(np.uint8)
        
        # Fast blend
        color_image = Image.fromarray(colored_array)
        blended = Image.blend(original_image, color_image, alpha=0.5)
        
        # Save path
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(blended_output_folder, f"{base_filename}_blended.png")
        
        # Save with optimization
        blended.save(save_path, optimize=True, compress_level=1)
        
        return save_path
        
    except Exception as e:
        logging.error(f"Fast blend failed for {image_path}: {e}")
        return None

def classify_images_ultra_fast(image_folder, band_output_folder, target_band=0, 
                              create_blended=False, blended_output_folder=None):
    """ULTRA-FAST image classification with GPU batch processing"""
    
    start_time = time.time()
    logging.info(f"🚀 TURBO CLASSIFICATION STARTED for: {image_folder}")
    
    # Prepare directories
    os.makedirs(band_output_folder, exist_ok=True)
    if create_blended and blended_output_folder:
        os.makedirs(blended_output_folder, exist_ok=True)

    # Get image files
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1].lower() in supported_extensions]

    if not image_files:
        logging.warning(f"No images found in {image_folder}")
        return {"blended_paths": [], "band_paths": []}

    # Load model once
    try:
        load_model_and_feature_extractor()
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return {"blended_paths": [], "band_paths": []}

    image_paths = [os.path.join(image_folder, f) for f in image_files]
    logging.info(f"📊 Processing {len(image_paths)} images with batch size {batch_size}")
    
    all_results = {"band_paths": [], "blended_paths": []}
    
    # Process in batches - TURBO MODE
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    with tqdm(total=len(image_paths), desc="🔥 TURBO Processing", unit="img") as pbar:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Classify batch - ULTRA FAST
            batch_results = classify_batch_ultra_fast(batch_paths)
            
            if batch_results:
                # Save results - OPTIMIZED
                saved_results = save_results_batch_optimized(
                    batch_results, band_output_folder, target_band, 
                    create_blended, blended_output_folder
                )
                
                all_results["band_paths"].extend(saved_results["band_paths"])
                all_results["blended_paths"].extend(saved_results["blended_paths"])
            
            pbar.update(len(batch_paths))
            
            # Memory cleanup every 3 batches
            if i % (batch_size * 3) == 0:
                gc.collect()
    
    # Final stats
    end_time = time.time()
    total_time = end_time - start_time
    speed = len(image_paths) / total_time if total_time > 0 else 0
    
    logging.info(f"🎯 TURBO COMPLETE!")
    logging.info(f"📊 Processed: {len(all_results['band_paths'])} band images")
    if create_blended:
        logging.info(f"🎨 Created: {len(all_results['blended_paths'])} blended images")
    logging.info(f"⏱️ Total time: {total_time:.2f}s")
    logging.info(f"🚀 Speed: {speed:.2f} images/second")
    
    return all_results

# Backward compatibility functions - TURBO VERSIONS
def process_images_band_only(image_folder="test_images", output_folder="band_output", target_band=0):
    """TURBO version of band-only processing"""
    if not os.path.exists(image_folder):
        logging.error(f"Image folder '{image_folder}' not found.")
        return {"band_paths": []}
    
    logging.info(f"🚀 TURBO processing: {image_folder}")
    logging.info(f"🎯 Target band: {['Red', 'Green', 'Blue'][target_band]}")
    
    results = classify_images_ultra_fast(
        image_folder=image_folder,
        band_output_folder=output_folder,
        target_band=target_band,
        create_blended=False
    )
    
    logging.info(f"✅ TURBO COMPLETE: {len(results['band_paths'])} images processed")
    return results

def classify_images_in_folder_with_band(image_folder, band_output_folder, target_band=0, 
                                      create_blended=False, blended_output_folder=None):
    """TURBO version with full compatibility"""
    return classify_images_ultra_fast(
        image_folder, band_output_folder, target_band, 
        create_blended, blended_output_folder
    )

def classify_images_in_folder(image_folder, blended_output_folder):
    """TURBO version for backward compatibility"""
    temp_band_folder = os.path.join(os.path.dirname(blended_output_folder), "temp_band")
    
    result = classify_images_ultra_fast(
        image_folder, 
        band_output_folder=temp_band_folder,
        create_blended=True, 
        blended_output_folder=blended_output_folder
    )
    
    # Clean up temp folder
    import shutil
    if os.path.exists(temp_band_folder):
        shutil.rmtree(temp_band_folder, ignore_errors=True)
    
    return result["blended_paths"]

def get_performance_info():
    """Get performance information"""
    setup_gpu_optimization()
    
    info = {
        "device": device_name,
        "gpu_available": gpu_available,
        "batch_size": batch_size,
        "mixed_precision": False,
        "turbo_mode": gpu_available
    }
    
    if gpu_available:
        try:
            policy = tf.keras.mixed_precision.global_policy()
            info["mixed_precision"] = policy.name == 'mixed_float16'
        except:
            pass
    
    return info

# Additional utility functions
def classify_image(image_path):
    """Single image classification - kept for compatibility"""
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

def save_segmentation_in_band(predicted_mask, original_image, original_size, output_path, target_band=0):
    """Legacy function - kept for compatibility"""
    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        mask_image = Image.fromarray(predicted_mask.astype(np.uint8))
        mask_resized = mask_image.resize(original_size, Image.NEAREST)
        mask_array = np.array(mask_resized)
        
        original_array = np.array(original_image.convert('RGB'))
        modified_image_array = original_array.copy()
        
        normalized_classes = (mask_array * (255 / 149)).astype(np.uint8)
        modified_image_array[:, :, target_band] = normalized_classes
        
        output_image = Image.fromarray(modified_image_array)
        output_image.save(output_path, optimize=True)
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error saving segmentation in band: {e}")
        return None

def create_colored_mask_and_blend(original_image_path, raw_mask, blended_output_folder):
    """Legacy function - kept for compatibility"""
    return create_colored_mask_and_blend_fast(original_image_path, raw_mask, blended_output_folder)

def save_band_modified_image(original_image_path, raw_mask, band_output_folder, target_band=0):
    """Legacy function - kept for compatibility"""
    return save_band_modified_image_fast(original_image_path, raw_mask, band_output_folder, target_band)

if __name__ == "__main__":
    # Performance info
    print("=== TURBO PERFORMANCE INFO ===")
    perf_info = get_performance_info()
    for key, value in perf_info.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # TURBO test
    print("=== TURBO TEST ===")
    process_images_band_only(
        image_folder="test_images",
        output_folder="turbo_output",
        target_band=0
    )
