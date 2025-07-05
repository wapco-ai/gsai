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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# Global variables
model = None
feature_extractor = None
cpu_threads = multiprocessing.cpu_count()

def setup_cpu_optimization():
    """Aggressive CPU optimization for maximum performance"""
    try:
        # Set CPU threads for maximum performance
        tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(cpu_threads // 2)
        
        # Enable CPU optimizations
        tf.config.optimizer.set_jit(True)  # XLA compilation
        
        logging.info(f"💻 CPU TURBO MODE ACTIVATED!")
        logging.info(f"🔧 CPU Cores: {cpu_threads}")
        logging.info(f"⚡ Intra-op threads: {cpu_threads}")
        logging.info(f"⚡ Inter-op threads: {cpu_threads // 2}")
        logging.info(f"🚀 XLA compilation: ENABLED")
        
        return True
        
    except Exception as e:
        logging.warning(f"CPU optimization failed: {e}")
        return False

def load_model_and_feature_extractor():
    """Load model with aggressive CPU optimization"""
    global model, feature_extractor
    
    if model is None or feature_extractor is None:
        logging.info("🚀 Loading model with CPU TURBO optimizations...")
        
        # Setup CPU optimization
        setup_cpu_optimization()
        
        try:
            # Load model
            if not os.path.exists(MODEL_DIR):
                logging.info("⬇ Downloading model...")
                model = TFSegformerForSemanticSegmentation.from_pretrained(MODEL_NAME, from_pt=True)
                model.save_pretrained(MODEL_DIR)
                logging.info("✅ Model downloaded and saved.")
            else:
                logging.info("📦 Loading model from local...")
                model = TFSegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
                logging.info("✅ Model loaded.")

            # Load feature extractor
            feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
            logging.info("✅ Feature extractor loaded.")
            
            logging.info(f"🔥 CPU TURBO MODE READY!")

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            model = None
            feature_extractor = None
            raise

def classify_image_cpu_optimized(image_path):
    """CPU-optimized single image classification"""
    if model is None or feature_extractor is None:
        load_model_and_feature_extractor()

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Prepare input
        inputs = feature_extractor(images=image, return_tensors="tf")
        
        # CPU inference with optimization
        outputs = model(**inputs)
        logits = outputs.logits

        # Get predictions
        predicted_mask = tf.argmax(logits, axis=1)[0].numpy()
        predicted_mask = np.where(predicted_mask > 149, 0, predicted_mask)

        # Clean up
        del inputs, outputs, logits
        gc.collect()

        return predicted_mask

    except Exception as e:
        logging.error(f"Error classifying image {image_path}: {e}")
        return None

def classify_batch_cpu_optimized(image_paths_batch, batch_size=2):
    """CPU-optimized batch processing with smaller batches"""
    if model is None or feature_extractor is None:
        load_model_and_feature_extractor()
    
    try:
        # Load images
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
        
        # Process in smaller sub-batches to avoid memory issues
        results = {}
        
        for i in range(0, len(images), batch_size):
            sub_batch_images = images[i:i + batch_size]
            sub_batch_paths = valid_paths[i:i + batch_size]
            
            # Prepare inputs
            inputs = feature_extractor(images=sub_batch_images, return_tensors="tf")
            
            # CPU inference
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Process results
            predicted_masks = tf.argmax(logits, axis=1).numpy()
            predicted_masks = np.where(predicted_masks > 149, 0, predicted_masks)
            
            # Store results
            for j, img_path in enumerate(sub_batch_paths):
                results[img_path] = predicted_masks[j]
            
            # Cleanup
            del inputs, outputs, logits, predicted_masks
            gc.collect()
        
        return results
        
    except Exception as e:
        logging.error(f"CPU batch classification failed: {e}")
        return {}

def save_results_parallel(results, band_output_folder, target_band=0, 
                         create_blended=False, blended_output_folder=None):
    """Parallel result saving using threading"""
    
    band_paths = []
    blended_paths = []
    
    # Ensure directories exist
    os.makedirs(band_output_folder, exist_ok=True)
    if create_blended and blended_output_folder:
        os.makedirs(blended_output_folder, exist_ok=True)
    
    def save_single_result(item):
        img_path, predicted_mask = item
        result = {"band_path": None, "blended_path": None}
        
        try:
            # Save band-modified image
            band_path = save_band_modified_image_fast(img_path, predicted_mask, band_output_folder, target_band)
            if band_path:
                result["band_path"] = band_path
            
            # Save blended image if requested
            if create_blended and blended_output_folder:
                blended_path = create_colored_mask_and_blend_fast(img_path, predicted_mask, blended_output_folder)
                if blended_path:
                    result["blended_path"] = blended_path
                    
        except Exception as e:
            logging.error(f"Failed to save results for {img_path}: {e}")
        
        return result
    
    # Use ThreadPoolExecutor for parallel I/O operations
    max_workers = min(4, cpu_threads // 2)  # Conservative thread count
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        save_results = list(executor.map(save_single_result, results.items()))
    
    # Collect results
    for result in save_results:
        if result["band_path"]:
            band_paths.append(result["band_path"])
        if result["blended_path"]:
            blended_paths.append(result["blended_path"])
    
    return {"band_paths": band_paths, "blended_paths": blended_paths}

def save_band_modified_image_fast(image_path, predicted_mask, band_output_folder, target_band=0):
    """Fast band modification saving"""
    try:
        # Load original image
        original_image = Image.open(image_path).convert("RGB")
        original_size = original_image.size
        original_array = np.array(original_image)
        
        # Resize mask
        mask_resized = Image.fromarray(predicted_mask.astype(np.uint8)).resize(original_size, Image.NEAREST)
        mask_array = np.array(mask_resized)
        
        # Vectorized normalization
        normalized_classes = (mask_array * (255 / 149)).astype(np.uint8)
        
        # Copy and modify
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
    """Fast blended image creation"""
    try:
        # Load original
        original_image = Image.open(image_path).convert("RGB")
        original_size = original_image.size
        
        # Resize mask
        mask_resized = Image.fromarray(raw_mask.astype(np.uint8)).resize(original_size, Image.NEAREST)
        resized_mask = np.array(mask_resized)
        
        # Vectorized coloring
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

def classify_images_cpu_turbo(image_folder, band_output_folder, target_band=0, 
                             create_blended=False, blended_output_folder=None):
    """CPU TURBO mode classification"""
    
    start_time = time.time()
    logging.info(f"💻 CPU TURBO CLASSIFICATION STARTED for: {image_folder}")
    
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
    logging.info(f"📊 Processing {len(image_paths)} images with CPU TURBO mode")
    
    all_results = {"band_paths": [], "blended_paths": []}
    
    # CPU batch size (smaller for memory efficiency)
    cpu_batch_size = 2
    
    # Process in batches
    with tqdm(total=len(image_paths), desc="💻 CPU TURBO Processing", unit="img") as pbar:
        for i in range(0, len(image_paths), cpu_batch_size):
            batch_paths = image_paths[i:i + cpu_batch_size]
            
            # Classify batch
            batch_results = classify_batch_cpu_optimized(batch_paths, batch_size=1)
            
            if batch_results:
                # Save results in parallel
                saved_results = save_results_parallel(
                    batch_results, band_output_folder, target_band, 
                    create_blended, blended_output_folder
                )
                
                all_results["band_paths"].extend(saved_results["band_paths"])
                all_results["blended_paths"].extend(saved_results["blended_paths"])
            
            pbar.update(len(batch_paths))
            
            # Memory cleanup
            if i % (cpu_batch_size * 5) == 0:
                gc.collect()
    
    # Final stats
    end_time = time.time()
    total_time = end_time - start_time
    speed = len(image_paths) / total_time if total_time > 0 else 0
    
    logging.info(f"💻 CPU TURBO COMPLETE!")
    logging.info(f"📊 Processed: {len(all_results['band_paths'])} band images")
    if create_blended:
        logging.info(f"🎨 Created: {len(all_results['blended_paths'])} blended images")
    logging.info(f"⏱️ Total time: {total_time:.2f}s")
    logging.info(f"🚀 Speed: {speed:.2f} images/second")
    
    return all_results

# Backward compatibility functions
def process_images_band_only(image_folder="test_images", output_folder="band_output", target_band=0):
    """CPU TURBO version of band-only processing"""
    if not os.path.exists(image_folder):
        logging.error(f"Image folder '{image_folder}' not found.")
        return {"band_paths": []}
    
    logging.info(f"💻 CPU TURBO processing: {image_folder}")
    logging.info(f"🎯 Target band: {['Red', 'Green', 'Blue'][target_band]}")
    
    results = classify_images_cpu_turbo(
        image_folder=image_folder,
        band_output_folder=output_folder,
        target_band=target_band,
        create_blended=False
    )
    
    logging.info(f"✅ CPU TURBO COMPLETE: {len(results['band_paths'])} images processed")
    return results

def classify_images_in_folder_with_band(image_folder, band_output_folder, target_band=0, 
                                      create_blended=False, blended_output_folder=None):
    """CPU TURBO version with full compatibility"""
    return classify_images_cpu_turbo(
        image_folder, band_output_folder, target_band, 
        create_blended, blended_output_folder
    )

def classify_images_in_folder(image_folder, blended_output_folder):
    """CPU TURBO version for backward compatibility"""
    temp_band_folder = os.path.join(os.path.dirname(blended_output_folder), "temp_band")
    
    result = classify_images_cpu_turbo(
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

# Legacy functions for compatibility
def classify_image(image_path):
    """Single image classification"""
    return classify_image_cpu_optimized(image_path)

def save_segmentation_in_band(predicted_mask, original_image, original_size, output_path, target_band=0):
    """Legacy function"""
    try:
        output_dir = os.path.dirname(output_path)
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
    """Legacy function"""
    return create_colored_mask_and_blend_fast(original_image_path, raw_mask, blended_output_folder)

def save_band_modified_image(original_image_path, raw_mask, band_output_folder, target_band=0):
    """Legacy function"""
    return save_band_modified_image_fast(original_image_path, raw_mask, band_output_folder, target_band)

def get_performance_info():
    """Get CPU performance information"""
    setup_cpu_optimization()
    
    info = {
        "device": "CPU",
        "cpu_cores": cpu_threads,
        "turbo_mode": True,
        "xla_enabled": True,
        "parallel_io": True
    }
    
    return info

if __name__ == "__main__":
    # Performance info
    print("=== CPU TURBO PERFORMANCE INFO ===")
    perf_info = get_performance_info()
    for key, value in perf_info.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # CPU TURBO test
    print("=== CPU TURBO TEST ===")
    process_images_band_only(
        image_folder="test_images",
        output_folder="cpu_turbo_output",
        target_band=0
    )
