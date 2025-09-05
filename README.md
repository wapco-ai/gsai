# GeoSphereAI

GeoSphereAI is a Flask application that processes images or video to generate 3D point clouds with Agisoft Metashape and performs semantic segmentation using NVIDIA SegFormer. The application supports common video formats such as MP4, AVI, MOV, MKV, and 360 files. This document explains how to install the project, configure Metashape and SpatiaLite, and run the app.

## Installation

1. Create and activate a Python 3.7 environment:
   ```bash
   conda create -n geosphereai python=3.7
   conda activate geosphereai
   ```
2. Install the required packages:
   ```bash
   conda install tensorflow-gpu=2.6.0 cudatoolkit=11.3 cudnn=8.2 -c conda-forge
   conda install opencv matplotlib numpy scipy scikit-learn pillow -c conda-forge
   conda install flask werkzeug tqdm transformers keras=2.6.0 -c conda-forge
   ```
3. (Optional) Verify that TensorFlow sees your GPU:
   ```bash
   python - <<'PY'
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   PY
   ```

## SpatiaLite setup

If you intend to store geospatial data, install SpatiaLite and create a new database:

```bash
sudo apt-get install spatialite-bin
spatialite geosphere.db < /usr/share/spatialite/init_spatialite.sql
```

Set the `SPATIALITE_DB` environment variable to the path of the database if your code uses it:

```bash
export SPATIALITE_DB=/path/to/geosphere.db
```

## Configuring Metashape

Edit `app.py` and set the `METASHAPE_EXECUTABLE` variable to the path of the Metashape binary:

```python
METASHAPE_EXECUTABLE = r"D:\\Program Files\\Agisoft\\Metashape Pro\\metashape.exe"
```

Adjust the path for your operating system.

## Environment variables

The application reads proxy settings from `HTTP_PROXY` and `HTTPS_PROXY`. You can set them manually if required:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### Using Open3D

The default environment for GeoSphereAI uses Python 3.7, which is too old for
`open3d`.  To enable Open3D features such as point‑cloud downsampling and
validation, install Open3D in a separate Python environment (e.g. Python 3.11)
and set the path to its interpreter via the `GEO3D_PY` environment variable:

```bash
export GEO3D_PY=/path/to/python-with-open3d
```

When `GEO3D_PY` is set, Open3D operations are executed through the helper
script `o3d_worker.py` in that external environment.

## Starting the application

Run the Flask server from the project directory:

```bash
python app.py
```

Visit `http://localhost:5000` in a browser to access the web interface.

## Database

All application data, including user accounts and process records, is stored in a
single SQLite file named `app.db` in the project directory. On first run the
application creates a default user `wapco` with password `wapco`.

Completed processes can be removed from the **Previous Processes** page.
Click the trash icon labeled **حذف پروژه** to delete a project along with its
attachments and analysis folder.

## Usage examples

Processing a directory of images from the command line:

```bash
metashape -r metashape_script.py --image_full_pipeline \
    --image_dir path/to/images --output_dir outputs/run1
```

By default the pipeline exports both PLY and PCD point clouds. Use
`--export_ply` and/or `--export_pcd` to control the formats:

```bash
metashape -r metashape_script.py --image_full_pipeline \
    --image_dir path/to/images --output_dir outputs/run1 --export_ply
```

The web interface performs similar commands internally when you upload files through the browser.
When uploading a video or ZIP file you can now choose the point cloud formats using checkboxes for **PLY** and **PCD**.

Use the new `--reference_preselection_mode` option to control Metashape's reference preselection strategy. Supported values are `source`, `estimated`, and `sequential`:

```bash
metashape -r metashape_script.py --image_full_pipeline \
    --image_dir path/to/images --output_dir outputs/run1 \
    --reference_preselection_mode sequential
```

The video and ZIP upload pages expose this setting through a dropdown labeled **حالت پیش‌انتخاب مرجع**.

Use `--sensor_type` to select the camera model. Allowed values are `Frame`, `Fisheye`, `Spherical`, and `Cylindrical`:

```bash
metashape -r metashape_script.py --image_full_pipeline \
    --image_dir path/to/images --output_dir outputs/run1 \
    --sensor_type Fisheye
```

The documentation for Metashape (user guide and Python API) is available in `static/docs/`.

## Optional analyses

The sign processing pipeline supports optional analysis steps. Use
`sign_pipeline.available_analyses()` to list the registered analyses and
pass the desired names to `process_sign_pipeline` via the `analyses`
parameter. For example:

```python
from sign_pipeline import process_sign_pipeline

process_sign_pipeline(
    image_dir="images",
    output_dir="outputs/run1",
    analyses=["sign_bbox"],
)
```

The provided `sign_bbox` analysis computes a bounding box around traffic
sign points (label `43`) and writes the results to `sign_bboxes.json` in
the output directory. When uploading a ZIP or video through the web
interface, available analyses appear under **تحلیل‌های اختیاری** so users
can select them without writing code.

