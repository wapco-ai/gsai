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
import sys
import logging
import shutil
from pathlib import Path
from typing import Any, Dict
from settings import (
    ENABLE_PLY_VALIDATION,
    PLY_ROUND_COORDS,
    PLY_ROUND_DECIMALS,
)

try:
    # Local helper to invoke Open3D in an external Python environment
    from open3d_bridge import run_o3d_worker
except Exception:  # pragma: no cover - bridge is optional at runtime
    run_o3d_worker = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def downsample_point_cloud(input_path, output_path, ratio=0.1, mode="ascii"):
    """Create a simplified preview of a point cloud using Open3D.

    Parameters
    ----------
    input_path : str
        Path to source point cloud.
    output_path : str
        Where to write the reduced point cloud.
    ratio : float
        Fraction of points to keep (0-1).
    mode : str
        ``"ascii"`` or ``"binary"`` output format.
    """
    try:
        import open3d as o3d
        import numpy as np

        if ratio <= 0 or ratio >= 1:
            raise ValueError("ratio must be between 0 and 1")

        pcd = o3d.io.read_point_cloud(input_path)
        if len(pcd.points) == 0:
            print("No points found in point cloud for downsampling")
            return False

        sampled = pcd.random_down_sample(ratio)
        points = np.asarray(sampled.points)
        sampled.points = o3d.utility.Vector3dVector(np.round(points, 2))
        o3d.io.write_point_cloud(output_path, sampled, write_ascii=(mode == "ascii"))
        print(f"Preview point cloud saved to {output_path}")
        return True
    except Exception as exc:
        if run_o3d_worker is None:
            print(f"Open3D not available for downsampling: {exc}")
            return False
        try:
            res: Dict[str, Any] = run_o3d_worker("downsample", input_path, output_path, ratio)
            if not res.get("ok"):
                print(f"Open3D worker failed: {res}")
                return False
            print(f"Preview point cloud saved to {output_path}")
            return True
        except Exception as worker_exc:
            print(f"Open3D worker not available: {worker_exc}")
            return False


def validate_ply_file(ply_path, mode=None, validate=ENABLE_PLY_VALIDATION):
    """Validate PLY header and ensure correct point count and format.

    Parameters
    ----------
    ply_path : str
        Path to PLY file.
    mode : str or None
        Desired format ("ascii" or "binary"). When ``None`` the existing
        file format is preserved.
    validate : bool
        Skip validation when ``False`` to improve performance.
    """
    if not validate:
        return True
    try:
        from plyfile import PlyData

        ply = PlyData.read(ply_path)
        vertex = ply['vertex']
        header_count = vertex.count
        actual_count = len(vertex.data)

        # Safely determine existing PLY format
        if isinstance(getattr(ply, "header", None), dict) and "format" in ply.header:
            ply_format = ply.header["format"][0]
        else:
            ply_format = "ascii" if getattr(ply, "text", False) else "binary_little_endian"

        if mode is None:
            mode = "ascii" if ply_format == "ascii" else "binary"
        target_text = mode == "ascii"
        expected_format = "ascii" if target_text else "binary_little_endian"

        # Rewrite file if header count or format is inconsistent
        needs_rewrite = header_count != actual_count or ply_format != expected_format
        if needs_rewrite:
            PlyData(ply.elements, text=target_text).write(ply_path)
            ply = PlyData.read(ply_path)
            vertex = ply['vertex']
            header_count = vertex.count
            actual_count = len(vertex.data)

        # Optional verification with Open3D
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(ply_path)
            if len(pcd.points) != actual_count:
                print(
                    f"Point count mismatch: header {actual_count}, Open3D {len(pcd.points)}"
                )
        except Exception as exc:
            if run_o3d_worker is not None:
                try:
                    res: Dict[str, Any] = run_o3d_worker("count", ply_path)
                    if res.get("ok") and res.get("points") != actual_count:
                        print(
                            f"Point count mismatch: header {actual_count}, Open3D {res.get('points')}"
                        )
                    elif not res.get("ok"):
                        print(f"Open3D worker error: {res}")
                except Exception as worker_exc:
                    print(f"Open3D validation skipped: {worker_exc}")
            else:
                print(f"Open3D validation skipped: {exc}")

        fields = vertex.data.dtype.names
        print(
            f"Validated PLY file {ply_path}: {actual_count} points, fields {list(fields)}"
        )
        return True
    except Exception as exc:
        print(f"Failed to validate PLY file {ply_path}: {exc}")
        return False
 
def add_class_field_to_ply(
    ply_path, mode=None, validate=ENABLE_PLY_VALIDATION, overwrite=False
):
    """Duplicate green channel into 'class' and 'label' fields in a PLY file.

    Parameters
    ----------
    ply_path : str or Path
        Input PLY file which will be rewritten.
    mode : str or None
        Output format, "ascii" or "binary". If ``None`` the input format is
        preserved.
    validate : bool
        Perform a full :func:`validate_ply_file` when True (default) but may
        slow processing.
    overwrite : bool
        When ``True`` rewrite ``ply_path`` in place instead of creating a new
        file with ``_with_class`` suffix.
    """
    try:
        import numpy as np
        from plyfile import PlyData, PlyElement

        ply_path = Path(ply_path)
        # Ensure PLY header is valid before processing
        validate_ply_file(str(ply_path), validate=validate)
        output_ply_path = (
            ply_path if overwrite else ply_path.with_name(ply_path.stem + '_with_class.ply')
        )

        plydata_in = PlyData.read(str(ply_path))
        if mode is None:
            mode = "ascii" if plydata_in.text else "binary"
        vertex = plydata_in['vertex']
        
        if 'green' not in vertex.data.dtype.names:
            print("No green channel found in PLY; skipping class field addition")
            return

        g = vertex['green']
        dtype_descr = list(vertex.data.dtype.descr)

        # این روش برای بررسی وجود فیلد امن‌تر است
        existing_fields = {name for name, fmt in dtype_descr}
        if 'class' not in existing_fields:
            dtype_descr.append(('class', 'u1'))
        if 'label' not in existing_fields:
            dtype_descr.append(('label', 'u1'))

        new_data = np.empty(vertex.count, dtype=dtype_descr)
        for name in vertex.data.dtype.names:
            new_data[name] = vertex.data[name]

        new_data['class'] = g
        new_data['label'] = g

        if PLY_ROUND_COORDS:
            for coord in ("x", "y", "z"):
                if coord in new_data.dtype.names:
                    new_data[coord] = np.round(
                        new_data[coord], PLY_ROUND_DECIMALS
                    )
        
        new_vertex_element = PlyElement.describe(new_data, 'vertex')

        other_elements = [el for el in plydata_in.elements if el.name != 'vertex']

        # Preserve the input format; set ``text=True`` above for ASCII output
        plydata_out = PlyData(
            [new_vertex_element] + other_elements,
            text=(mode == "ascii"),
        )

        # Write the output without forcing text mode
        output_ply_path = output_ply_path.resolve()
        plydata_out.write(str(output_ply_path))
        validate_ply_file(str(output_ply_path), mode, validate)
        print(f"SUCCESS: Added class field to PLY and saved to {output_ply_path}")

    except Exception as exc:
        import traceback
        print(f"Failed to add class field to PLY: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())


def add_class_field_to_pcd(pcd_path, mode=None):
    """
    Duplicate green channel into 'class' and 'label' fields in a PCD file.
    This version uses a safer method to add fields to prevent data corruption.
    """
    try:
        import numpy as np
        # تابع rfn دیگر استفاده نمی‌شود
        # from numpy.lib import recfunctions as rfn 
        from pypcd_imp import pypcd

        pc = pypcd.PointCloud.from_path(pcd_path)
        
        # ۱. استخراج کانال سبز (بدون تغییر)
        if 'green' in pc.fields:
            g = pc.pc_data['green'].astype(np.uint8)
        elif 'rgb' in pc.fields:
            rgb_bytes = pc.pc_data['rgb'].copy().view(np.uint8).reshape(-1, 4)
            g = rgb_bytes[:, 1].astype(np.uint8)
        else:
            print("No color information found in PCD; skipping class field addition")
            return

        # ۲. تعریف ساختار داده جدید (dtype)
        old_dtype = pc.pc_data.dtype.descr
        new_dtype = list(old_dtype) # کپی کردن ساختار قدیمی
        
        existing_fields = {name for name, fmt in old_dtype}
        if 'class' not in existing_fields:
            new_dtype.append(('class', 'u1')) # u1 = np.uint8
        if 'label' not in existing_fields:
            new_dtype.append(('label', 'u1'))

        # اگر هیچ فیلد جدیدی اضافه نشده باشد، نیازی به ادامه نیست
        if len(new_dtype) == len(old_dtype):
            print("Fields 'class' and 'label' already exist. Skipping.")
            # فایل اصلی را در مسیر خروجی کپی می‌کنیم تا گردش کار ادامه یابد
            output_pcd_path = pcd_path.replace('.pcd', '_with_class.pcd')
            shutil.copy(pcd_path, output_pcd_path)
            return

        # ۳. ایجاد آرایه خالی با ساختار جدید
        new_pc_data = np.empty(pc.pc_data.shape, dtype=new_dtype)

        # ۴. کپی کردن داده‌ها به صورت ستون به ستون (روش امن)
        for name in pc.pc_data.dtype.names:
            new_pc_data[name] = pc.pc_data[name]
        
        # ۵. پر کردن ستون‌های جدید
        if 'class' in [n for n, f in new_dtype]:
            new_pc_data['class'] = g
        if 'label' in [n for n, f in new_dtype]:
            new_pc_data['label'] = g

        # ۶. جایگزین کردن داده‌های قدیمی با داده‌های جدید
        pc.pc_data = new_pc_data
        
        # متادیتا به صورت خودکار توسط save_pcd به‌روز می‌شود

        output_pcd_path = pcd_path.replace('.pcd', '_with_class.pcd')
        compression = 'ascii' if mode == 'ascii' else 'binary'
        pc.save_pcd(output_pcd_path, compression=compression)
        
        print(f"SUCCESS (Safe Mode): Added class field to PCD and saved to {output_pcd_path}")

    except Exception as exc:
        import traceback
        print(f"Failed to add class field to PCD: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())


# ----------------------------  
# Frame Extraction Functions  
# ----------------------------  

def extract_frames(video_path, output_dir, start_time=0, end_time=None, frame_interval=1, crop_height_ratio=0.1):  
    import cv2  
    from tqdm import tqdm  

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS) or 29.97
    print(f"Using frame rate: {fps} FPS.")

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
    reference_file=None,
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

    if reference_file:
        try:
            chunk.importReference(
                path=reference_file,
                format=Metashape.ReferenceFormatCSV,
                delimiter=",",
            )
            chunk.crs = Metashape.CoordinateSystem("EPSG::4326")
            chunk.updateTransform() # for update and aply the refrence
            print(f"Reference data imported from {reference_file}")
        except Exception as e:
            print(f"Failed to import reference file: {e}")

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
        keypoint_limit=50000,
        tiepoint_limit=5000
    )
    chunk.alignCameras()  

    # Build depth maps  
    if chunk.cameras:  
        chunk.buildDepthMaps(downscale=2)  
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

def convert_to_point_cloud(
    project_path,
    output_dir,
    export_ply=True,
    export_pcd=True,
    export_potree=False,
    ply_mode="binary",
    pcd_mode="binary",
    ply_preview_pct=None,
    pcd_preview_pct=None,
    validate=ENABLE_PLY_VALIDATION,
):
    import Metashape

    #print(f"\nMetashape ver: {Metashape.app.version}")

    # Load the project
    doc = Metashape.Document()
    doc.open(project_path)
    chunk = doc.chunk

    export_crs = chunk.crs

    def has_reference(chunk_obj):
        try:
            if any(
                getattr(cam.reference, "location", None) is not None
                for cam in chunk_obj.cameras
            ):
                return True
        except Exception:
            pass
        try:
            if any(
                getattr(marker.reference, "location", None) is not None
                for marker in chunk_obj.markers
            ):
                return True
        except Exception:
            pass
        return False

    if export_crs and has_reference(chunk):
        is_geo = False
        try:
            is_geo = bool(getattr(export_crs, "isGeographic", False))
        except Exception:
            pass
        if not is_geo:
            wkt = getattr(export_crs, "wkt", "")
            if isinstance(wkt, str):
                is_geo = "GEOGCS" in wkt.upper()
        if is_geo:
            export_crs = Metashape.CoordinateSystem("EPSG::32640")

    crs_params = {"crs": export_crs} if export_crs else {}

    # Check if dense point cloud exists  
    if chunk.point_cloud is None:  
        print("No dense point cloud found. Building dense point cloud...")  

        # Build depth maps (if not already built)  
        if not chunk.depth_maps:  
            print("Building depth maps...")  
            chunk.buildDepthMaps(downscale=2)  
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
        ply_paths = []
        if export_ply:
            if ply_mode in ("binary", "both"):
                output_path = os.path.join(output_dir, "point_cloud.ply")
                chunk.exportPointCloud(
                    output_path,
                    format=Metashape.PointCloudFormatPLY,
                    binary=True,
                    save_point_classification=True,
                    save_point_color=True,
                    **crs_params,
                )
                if add_class_field:
                    add_class_field_to_ply(
                        output_path, mode="binary", validate=validate, overwrite=True
                    )
                ply_paths.append((output_path, "binary"))
                print(f"ply Point cloud exported to {output_path}")
            if ply_mode in ("ascii", "both"):
                name = "point_cloud_ascii.ply" if ply_mode == "both" else "point_cloud.ply"
                output_path = os.path.join(output_dir, name)
                chunk.exportPointCloud(
                    output_path,
                    format=Metashape.PointCloudFormatPLY,
                    binary=False,
                    save_point_classification=True,
                    save_point_color=True,
                    **crs_params,
                )
                if add_class_field:
                    add_class_field_to_ply(
                        output_path, mode="ascii", validate=validate, overwrite=True
                    )
                ply_paths.append((output_path, "ascii"))
                print(f"ply Point cloud exported to {output_path}")

        pcd_paths = []
        if export_pcd:
            if pcd_mode in ("binary", "both"):
                output_path = os.path.join(output_dir, "point_cloud.pcd")
                chunk.exportPointCloud(
                    output_path,
                    format=Metashape.PointCloudFormatPCD,
                    binary=True,
                    save_point_color=True,
                    **crs_params,
                )
                if add_class_field:
                    add_class_field_to_pcd(output_path, mode="binary")
                pcd_paths.append((output_path, "binary"))
                print(f"pcd Point cloud exported to {output_path}")
            if pcd_mode in ("ascii", "both"):
                name = "point_cloud_ascii.pcd" if pcd_mode == "both" else "point_cloud.pcd"
                output_path = os.path.join(output_dir, name)
                chunk.exportPointCloud(
                    output_path,
                    format=Metashape.PointCloudFormatPCD,
                    binary=False,
                    save_point_color=True,
                    **crs_params,
                )
                if add_class_field:
                    add_class_field_to_pcd(output_path, mode="ascii")
                pcd_paths.append((output_path, "ascii"))
                print(f"pcd Point cloud exported to {output_path}")
            
        if export_potree:
            potree_dir = os.path.join(output_dir, "potree")
            if os.path.exists(potree_dir):
                shutil.rmtree(potree_dir)
            chunk.exportPointCloud(
                potree_dir,
                format=Metashape.PointCloudFormatPotree,
                save_point_classification=True,
                save_point_color=True,
                **crs_params,
            )

            cloud_js = os.path.join(potree_dir, "cloud.js")
            data_dirs = [
                d
                for d in os.listdir(potree_dir)
                if os.path.isdir(os.path.join(potree_dir, d)) and d.lower().startswith("data")
            ]

            if not os.path.exists(cloud_js) or not data_dirs:
                missing = []
                if not os.path.exists(cloud_js):
                    missing.append("cloud.js")
                if not data_dirs:
                    missing.append("data folder")
                logging.error(
                    f"Incomplete Potree export at {potree_dir}: missing {', '.join(missing)}"
                )
            else:
                print(f"Potree point cloud exported to {potree_dir}")

        if ply_preview_pct:
            ratio = float(ply_preview_pct) / 100.0
            for path, pmode in ply_paths:
                preview_path = path.replace('.ply', '_preview.ply')
                if downsample_point_cloud(path, preview_path, ratio, pmode):
                    if add_class_field:
                        add_class_field_to_ply(
                            preview_path, mode=pmode, validate=validate, overwrite=True
                        )
        if pcd_preview_pct:
            ratio = float(pcd_preview_pct) / 100.0
            for path, pmode in pcd_paths:
                preview_pcd = path.replace('.pcd', '_preview.pcd')
                if downsample_point_cloud(path, preview_pcd, ratio, pmode):
                    if add_class_field:
                        add_class_field_to_pcd(preview_pcd, mode=pmode)
                        preview_pcd_with_class = preview_pcd.replace('.pcd', '_with_class.pcd')
                        if os.path.exists(preview_pcd_with_class):
                            os.replace(preview_pcd_with_class, preview_pcd)


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
    export_ply = "--export_ply" in sys.argv
    export_pcd = "--export_pcd" in sys.argv
    export_potree = "--export_potree" in sys.argv
    add_class_field = "--add_class_field" in sys.argv
    validate = "--validate_ply" in sys.argv or ENABLE_PLY_VALIDATION  # integrity check
    ply_mode = sys.argv[sys.argv.index("--ply-mode") + 1] if "--ply-mode" in sys.argv else "binary"
    pcd_mode = sys.argv[sys.argv.index("--pcd-mode") + 1] if "--pcd-mode" in sys.argv else "binary"
    ply_preview_pct = float(sys.argv[sys.argv.index("--ply-preview-pct") + 1]) if "--ply-preview-pct" in sys.argv else None
    pcd_preview_pct = float(sys.argv[sys.argv.index("--pcd-preview-pct") + 1]) if "--pcd-preview-pct" in sys.argv else None
    if not export_ply and not export_pcd and not export_potree:
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
    reference_file = (
        sys.argv[sys.argv.index("--reference_file") + 1]
        if "--reference_file" in sys.argv
        else None
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
            image_dir, output_dir, reference_preselection_mode, sensor_type, reference_file
        )

    if "--convert_to_point_cloud" in sys.argv:  
        # Convert to point cloud directly  
        project_path = sys.argv[sys.argv.index("--convert_to_point_cloud") + 1]  
        output_dir = sys.argv[sys.argv.index("--output_dir") + 1]
        convert_to_point_cloud(
            project_path,
            output_dir,
            export_ply,
            export_pcd,
            export_potree,
            ply_mode,
            pcd_mode,
            ply_preview_pct,
            pcd_preview_pct,
            validate,
        )
    
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
            image_dir, output_dir, reference_preselection_mode, sensor_type, reference_file
        )
        
        # Convert to point cloud  
        project_path = os.path.join(output_dir, "project.psx")
        convert_to_point_cloud(
            project_path,
            output_dir,
            export_ply,
            export_pcd,
            export_potree,
            ply_mode,
            pcd_mode,
            ply_preview_pct,
            pcd_preview_pct,
            validate,
        )
        
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
            image_dir, metashape_output_dir, reference_preselection_mode, sensor_type, reference_file
        )

        # Convert to point cloud  
        project_path = os.path.join(metashape_output_dir, "project.psx")
        convert_to_point_cloud(
            project_path,
            metashape_output_dir,
            export_ply,
            export_pcd,
            export_potree,
            ply_mode,
            pcd_mode,
            ply_preview_pct,
            pcd_preview_pct,
            validate,
        )
