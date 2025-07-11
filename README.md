# GeoSphereAI

GeoSphereAI is a Flask application that processes images or video to generate 3D point clouds with Agisoft Metashape and performs semantic segmentation using NVIDIA SegFormer. This document explains how to install the project, configure Metashape and SpatiaLite, and run the app.

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

