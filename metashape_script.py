'''
cd "D:\Program Files\Agisoft\Metashape Pro"  
metashape -r "D:\AI\3dRecognition\pycode\metashaspe-v2.py" --image_dir "D:\AI\3dRecognition\output_metashape\frames" --output_dir "D:\AI\3dRecognition\output_metashape"
python script.py --extract_frames "path/to/video.mp4" --image_dir "path/to/output_frames" --start_time 150 --end_time 180 --frame_interval 0.3 --crop_height_ratio 0.01
python script.py --process_in_metashape --image_dir "path/to/images" --output_dir "path/to/output"
metashape -r "D:\AI\3dRecognition\pycode\metashaspe-v3.py" --convert_to_point_cloud "D:\AI\3dRecognition\output_metashape\project.psx" --output_dir "D:\AI\3dRecognition\output_metashape\output"
metashape -r metashape_script.py --image_full_pipeline --image_dir images --output_dir out --export_ply
python metashape_script.py --video_full_pipeline "D:/AI/3dRecognition/torghabe/torghabe.mp4" --output_dir "D:/AI/3dRecognition/output_metashape/torghabe" --start_time 0 --end_time 240 --frame_interval 1 --crop_height_ratio 0.01
metashape -r "D:\AI\3dRecognition\pycode\metashaspe-v3.py" --create_and_export_3d_model "D:\AI\3dRecognition\output_metashape\project.psx" --output_dir "D:\AI\3dRecognition\output_metashape" --model_format "obj"
'''  
import os
import subprocess
import sys


def downsample_point_cloud(input_path, output_path, ratio=0.1):
    """Create a simplified preview of a point cloud using Open3D."""
    try:
        import open3d as o3d
    except Exception as exc:
        print(f"Open3D not available for downsampling: {exc}")
        return False

    if ratio <= 0 or ratio >= 1:
        raise ValueError("ratio must be between 0 and 1")

    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        print("No points found in point cloud for downsampling")
        return False

    sampled = pcd.random_down_sample(ratio)
    o3d.io.write_point_cloud(output_path, sampled)
    print(f"Preview point cloud saved to {output_path}")
    return True

# ----------------------------  
# Frame Extraction Functions  
# ----------------------------  

def extract_frames(video_path, output_dir, start_time=0, end_time=None, frame_interval=1, crop_height_ratio=0.1):  
    import cv2  
    from tqdm import tqdm  

    fps = 29.97  
    print(f"Using frame rate: {fps} FPS.")  

    video = cv2.VideoCapture(video_path)  
    if not video.isOpened():  
        raise ValueError(f"Could not open video file: {video_path}")  

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  
    print(f"Total frames in video: {total_frames}")  

    if end_time is None:  
        end_time = total_frames / fps  

    start_frame = int(start_time * fps)  
    end_frame = int(end_time * fps)  
    frame_interval_in_frames = int(frame_interval * fps)  

    extracted_frame_indices = range(start_frame, end_frame, frame_interval_in_frames)  

    with tqdm(total=len(extracted_frame_indices), desc="Extracting Frames", unit="frame") as pbar:  
        for i in extracted_frame_indices:  
            video.set(cv2.CAP_PROP_POS_FRAMES, i)  
            ret, frame = video.read()
            if ret:
                height = frame.shape[0]
                crop_pixels = int(height * crop_height_ratio)
                cropped_frame = frame[:-crop_pixels, :, :] if crop_pixels > 0 else frame
                output_filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(output_filename, cropped_frame)
                pbar.update(1)  
            else:  
                print(f"Warning: Could not read frame {i}")  
                pbar.update(1)  

    video.release()  
    print("Frame extraction completed.")  
    return fps  

# --------------------------  
# Metashape Processing Functions  
# --------------------------  

def process_in_metashape(
    image_dir,
    output_dir,
    reference_preselection_mode="source",
    sensor_type="Frame",
):
    import Metashape
    
    #print(f"\nMetashape ver: {Metashape.app.version}")  
    
    doc = Metashape.Document()  
    chunk = doc.addChunk()  

    # Add images
    chunk.addPhotos(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tif",
                    ".tiff",
                    ".JPG",
                    ".JPEG",
                    ".PNG",
                    ".TIF",
                    ".TIFF",
                )
            )
        ]
    )
    print(f"Loaded {len(chunk.cameras)} images")

    # Set sensor type for all sensors
    type_map = {
        "Frame": Metashape.Sensor.Type.Frame,
        "Fisheye": Metashape.Sensor.Type.Fisheye,
        "Spherical": Metashape.Sensor.Type.Spherical,
        "Cylindrical": Metashape.Sensor.Type.Cylindrical,
    }
    sensor_enum = type_map.get(sensor_type, Metashape.Sensor.Type.Frame)
    for sensor in chunk.sensors:
        sensor.type = sensor_enum

    # Camera alignment  
    mode_map = {
        "source": Metashape.ReferencePreselectionSource,
        "estimated": Metashape.ReferencePreselectionEstimated,
        "sequential": Metashape.ReferencePreselectionSequential,
    }
    ref_mode = mode_map.get(reference_preselection_mode, Metashape.ReferencePreselectionSource)
    chunk.matchPhotos(
        downscale=2,
        generic_preselection=True,
        reference_preselection=True,
        reference_preselection_mode=ref_mode,
        keypoint_limit=1000000,
        tiepoint_limit=100000
    )
    chunk.alignCameras()  

    # Build depth maps  
    if chunk.cameras:  
        chunk.buildDepthMaps(downscale=1)  
        print("Depth maps built successfully!")  

        # Build dense cloud  
        if hasattr(chunk, 'buildPointCloud'):  
            chunk.buildPointCloud()  
            print("Dense cloud built successfully!")  
        else:  
            print("No valid method to build dense cloud found.")  

    else:  
        print("No cameras available in the chunk.")  

    doc.save(os.path.join(output_dir, "project.psx"))  
    print("Reconstruction completed ✓")  

def convert_to_point_cloud(project_path, output_dir, preview_ratio=None, export_ply=True, export_pcd=True):
    import Metashape  

    #print(f"\nMetashape ver: {Metashape.app.version}")  

    # Load the project  
    doc = Metashape.Document()  
    doc.open(project_path)  
    chunk = doc.chunk  

    # Check if dense point cloud exists  
    if chunk.point_cloud is None:  
        print("No dense point cloud found. Building dense point cloud...")  

        # Build depth maps (if not already built)  
        if not chunk.depth_maps:  
            print("Building depth maps...")  
            chunk.buildDepthMaps(downscale=1)  
            print("Depth maps built successfully.")  

        # Build point cloud  
        print("Building point cloud...")  
        chunk.buildPointCloud()  
        if chunk.point_cloud:  
            print(f"point cloud built successfully with {chunk.point_cloud.point_count} points.")
            doc.save(os.path.join(output_dir, "project.psx"))  
            print("point cloud saved to project completed ✓")  
        else:  
            print("Failed to build dense cloud.")  
            return  # Exit if there's no dense point cloud  

    # Export point cloud
    try:
        if export_ply:
            output_path = os.path.join(output_dir, "point_cloud.ply")
            chunk.exportPointCloud(
                output_path,
                format=Metashape.PointCloudFormatPLY,  # Point cloud format (PLY)
                crs=chunk.crs,  # Coordinate Reference System
                binary=True,
                save_point_classification=True
            )
            print(f"ply Point cloud exported to {output_path}")

        if export_pcd:
            output_path = os.path.join(output_dir, "point_cloud.pcd")
            chunk.exportPointCloud(
                output_path,
                format=Metashape.PointCloudFormatPCD,  # Point cloud format (pcd)
                crs=chunk.crs,  # Coordinate Reference System
                binary=True
            )
            print(f"pcd Point cloud exported to {output_path}")

        if preview_ratio:
            if export_ply:
                ply_path = os.path.join(output_dir, "point_cloud.ply")
                preview_path = os.path.join(output_dir, "point_cloud_preview.ply")
                if os.path.exists(ply_path):
                    downsample_point_cloud(ply_path, preview_path, preview_ratio)
            if export_pcd:
                pcd_path = os.path.join(output_dir, "point_cloud.pcd")
                preview_pcd = os.path.join(output_dir, "point_cloud_preview.pcd")
                if os.path.exists(pcd_path):
                    downsample_point_cloud(pcd_path, preview_pcd, preview_ratio)
    
    except Exception as e:  
        print(f"Error exporting point cloud: {e}") 

# --------------------------  
# Main Execution Flow  
# --------------------------  
def create_and_export_3d_model(project_path, output_dir, model_format="obj"):  
    import Metashape  

    #print(f"\nMetashape ver: {Metashape.app.version}")  

    # Load the project  
    doc = Metashape.Document()  
    doc.open(project_path)  
    chunk = doc.chunk  

    # Check if mesh exists, if not, create it  
    if chunk.model is None:  
        print("No 3D model (mesh) found. Creating a new mesh from the dense cloud...")  

        # Ensure that a dense cloud exists in the project  
        '''if chunk.point_cloud is None:  
            print("Error: Dense point cloud not found. Build a dense cloud before creating a mesh.")  
            return  '''

        # Build the model (mesh) from the dense cloud  
        try:  
            chunk.buildModel(surface_type=Metashape.Arbitrary,  
                             interpolation=Metashape.EnabledInterpolation,  
                             face_count=Metashape.MediumFaceCount)  
            if chunk.model:  
                print("3D model (mesh) created successfully.")
                doc.save(os.path.join(output_dir, "project.psx"))  
                print("model saved to project completed ✓")  
            else:  
                print("Failed to create 3D model (mesh).")  
                return  
        except Exception as e:  
            print(f"Error building 3D model: {e}")  
            return  

    # Export the 3D model  
    supported_formats = ("obj", "ply", "3ds")  
    if model_format.lower() not in supported_formats:  
        raise ValueError(f"Unsupported format '{model_format}'. Supported formats: {supported_formats}")  

    output_path = os.path.join(output_dir, f"metashape_3d_model.{model_format.lower()}")  
    try:  
        chunk.exportModel(output_path,  
                          binary=True,  
                          precision=6,  
                          texture_format=Metashape.ImageFormatJPEG if model_format.lower() == "obj" else None,  
                          save_texture=True,
                          save_colors=True,  
                          #comment="generated by wapco",  
                          save_normals=True)  
        print(f"3D model exported successfully to {output_path}")  
    except Exception as e:  
        print(f"Error exporting 3D model: {e}")
# --------------------------  
# Main Execution Flow  
# --------------------------  

if __name__ == "__main__":
    preview_ratio = float(sys.argv[sys.argv.index("--preview_ratio") + 1]) if "--preview_ratio" in sys.argv else None
    export_ply = "--export_ply" in sys.argv
    export_pcd = "--export_pcd" in sys.argv
    if not export_ply and not export_pcd:
        export_ply = export_pcd = True
    reference_preselection_mode = (
        sys.argv[sys.argv.index("--reference_preselection_mode") + 1]
        if "--reference_preselection_mode" in sys.argv
        else "source"
    )
    sensor_type = (
        sys.argv[sys.argv.index("--sensor_type") + 1]
        if "--sensor_type" in sys.argv
        else "Frame"
    )
    if "--extract_frames" in sys.argv:  
        # Extract frames from video  
        video_path = sys.argv[sys.argv.index("--extract_frames") + 1]  
        image_dir = sys.argv[sys.argv.index("--image_dir") + 1]  
        start_time = float(sys.argv[sys.argv.index("--start_time") + 1]) if "--start_time" in sys.argv else 0  
        end_time = float(sys.argv[sys.argv.index("--end_time") + 1]) if "--end_time" in sys.argv else None  
        frame_interval = float(sys.argv[sys.argv.index("--frame_interval") + 1]) if "--frame_interval" in sys.argv else 1  
        crop_height_ratio = float(sys.argv[sys.argv.index("--crop_height_ratio") + 1]) if "--crop_height_ratio" in sys.argv else 0.1  

        extract_frames(video_path, image_dir, start_time, end_time, frame_interval, crop_height_ratio)  

    if "--process_in_metashape" in sys.argv:
        # Process images in Metashape
        image_dir = sys.argv[sys.argv.index("--image_dir") + 1]
        output_dir = sys.argv[sys.argv.index("--output_dir") + 1]
        process_in_metashape(
            image_dir, output_dir, reference_preselection_mode, sensor_type
        )

    if "--convert_to_point_cloud" in sys.argv:  
        # Convert to point cloud directly  
        project_path = sys.argv[sys.argv.index("--convert_to_point_cloud") + 1]  
        output_dir = sys.argv[sys.argv.index("--output_dir") + 1]
        convert_to_point_cloud(project_path, output_dir, preview_ratio, export_ply, export_pcd)
    
    if "--create_and_export_3d_model" in sys.argv:  
        project_path = sys.argv[sys.argv.index("--create_and_export_3d_model") + 1]  
        output_dir = sys.argv[sys.argv.index("--output_dir") + 1]  
        model_format = sys.argv[sys.argv.index("--model_format") + 1] if "--model_format" in sys.argv else "obj"  
        create_and_export_3d_model(project_path, output_dir, model_format)
    
    if "--image_full_pipeline" in sys.argv:
        # Process images in Metashape
        image_dir = sys.argv[sys.argv.index("--image_dir") + 1]
        output_dir = sys.argv[sys.argv.index("--output_dir") + 1]
        process_in_metashape(
            image_dir, output_dir, reference_preselection_mode, sensor_type
        )
        
        # Convert to point cloud  
        project_path = os.path.join(output_dir, "project.psx")
        convert_to_point_cloud(project_path, output_dir, preview_ratio, export_ply, export_pcd)
        
    if "--video_full_pipeline" in sys.argv:  
        # Run the full pipeline (extract frames → process in Metashape → export point cloud)  
        video_path = sys.argv[sys.argv.index("--video_full_pipeline") + 1]  
        output_base_dir = sys.argv[sys.argv.index("--output_dir") + 1]  
        start_time = float(sys.argv[sys.argv.index("--start_time") + 1]) if "--start_time" in sys.argv else 0  
        end_time = float(sys.argv[sys.argv.index("--end_time") + 1]) if "--end_time" in sys.argv else None  
        frame_interval = float(sys.argv[sys.argv.index("--frame_interval") + 1]) if "--frame_interval" in sys.argv else 1  
        crop_height_ratio = float(sys.argv[sys.argv.index("--crop_height_ratio") + 1]) if "--crop_height_ratio" in sys.argv else 0.1  

        # Setup directories  
        os.makedirs(output_base_dir, exist_ok=True)  
        image_dir = os.path.join(output_base_dir, "frames")  
        os.makedirs(image_dir, exist_ok=True)  

        # Extract frames  
        extract_frames(  
            video_path=video_path,  
            output_dir=image_dir,  
            start_time=start_time,  
            end_time=end_time,  
            frame_interval=frame_interval,  
            crop_height_ratio=crop_height_ratio  
        )  

        # Process in Metashape  
        metashape_output_dir = os.path.join(output_base_dir, "project")  
        process_in_metashape(
            image_dir, metashape_output_dir, reference_preselection_mode, sensor_type
        )

        # Convert to point cloud  
        project_path = os.path.join(metashape_output_dir, "project.psx")
        convert_to_point_cloud(project_path, metashape_output_dir, preview_ratio, export_ply, export_pcd)
