import os
import zipfile
import logging
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    flash,
    send_from_directory,
    session,
)
from werkzeug.utils import secure_filename
import subprocess
import uuid
import json
from threading import Thread
import sys  # Import sys to get the python executable

if sys.platform.startswith("win"):
    import winreg
else:
    winreg = None


# Import the new image classifier module
import image_classifier


def apply_windows_proxy():
    if not sys.platform.startswith("win") or winreg is None:
        return
    try:
        # باز کردن کلید تنظیمات اینترنت کاربر جاری
        reg_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
            proxy_enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if proxy_enable:
                # اگر پروکسی فعال است، مقدار ProxyServer را بخوان
                proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
                # ست کردن متغیرهای محیطی
                os.environ["HTTP_PROXY"] = proxy_server
                os.environ["HTTPS_PROXY"] = proxy_server
                print(f"Proxy enabled: {proxy_server}")
            else:
                # اگر پروکسی غیرفعال است، مطمئن شو که متغیرها حذف شده‌اند
                os.environ.pop("HTTP_PROXY", None)
                os.environ.pop("HTTPS_PROXY", None)
                print("Proxy disabled, using direct connection.")
    except OSError as e:
        print(f"Failed to read Windows proxy settings: {e}")


# استفاده:
apply_windows_proxy()

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = {"jpg", "jpeg", "png"}
app.config["ALLOWED_VIDEO_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv"}
app.config["ALLOWED_ZIP_EXTENSIONS"] = {"zip"}
app.secret_key = "your_secret_key"  # CHANGE THIS TO A REAL, SECRET KEY
# Helper function to load Metashape executable path
app.config["PROCESSING_STATES"] = {}
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///processes.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class Process(db.Model):
    id = db.Column(db.String, primary_key=True)
    filename = db.Column(db.String)
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    duration = db.Column(db.Float)
    frame_count = db.Column(db.Integer)
    user = db.Column(db.String)
    status = db.Column(db.String)
    output_folder = db.Column(db.String)
    progress = db.Column(db.Integer, default=0)
    message = db.Column(db.String)


with app.app_context():
    db.create_all()


def load_metashape_executable():
    env_path = os.environ.get("METASHAPE_EXECUTABLE")
    if env_path:
        return env_path

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as cfg:
                data = json.load(cfg)
            if data.get("METASHAPE_EXECUTABLE"):
                return data["METASHAPE_EXECUTABLE"]
        except Exception as exc:
            raise RuntimeError(f"Failed to load config file {config_path}: {exc}")

    raise RuntimeError(
        "METASHAPE_EXECUTABLE not set in environment or config file"
    )


METASHAPE_EXECUTABLE = load_metashape_executable()
# Assuming metashape_script.py is in the same directory as app.py
METASHAPE_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "metashape_script.py"
)

# Ensure directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Process(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    process_uuid = db.Column(db.String(36), unique=True)
    filename = db.Column(db.String(255))
    user = db.Column(db.String(50))
    frame_count = db.Column(db.Integer)
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    duration = db.Column(db.Float)
    status = db.Column(db.String(50))


# Helper function to check allowed files
def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


# Helper function to extract ZIP files
def extract_images_from_zip(zip_path, output_folder):
    """Extract images from a ZIP file to the output folder."""
    extracted_files_count = 0  # Count of successfully extracted images
    allowed_extensions = app.config["ALLOWED_IMAGE_EXTENSIONS"]
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file in zip_ref.namelist():
                if "../" in file or file.startswith("/"):
                    logging.warning(
                        f"Skipping potentially malicious path in zip: {file}"
                    )
                    continue

                if allowed_file(file, allowed_extensions):
                    try:
                        # Construct the full path for extraction target
                        extract_target_path = os.path.join(output_folder, file)
                        # Normalize path to check against output_folder
                        normalized_extract_target = os.path.normpath(
                            extract_target_path
                        )

                        # Ensure extraction is within the intended directory
                        if normalized_extract_target.startswith(
                            os.path.normpath(output_folder)
                        ):
                            zip_ref.extract(file, output_folder)
                            extracted_files_count += 1
                        else:
                            logging.warning(
                                f"Skipping extraction outside target directory: {file}"
                            )

                    except Exception as e:
                        logging.error(f"Error extracting file {file} from zip: {e}")

    except zipfile.BadZipFile:
        logging.error(f"Invalid ZIP file: {zip_path}")
        flash("Invalid ZIP file.")
    except Exception as e:
        logging.error(f"Error processing ZIP file {zip_path}: {e}")
        flash(f"Error processing ZIP file: {e}")

    return extracted_files_count


# Helper function to update process state both in-memory and in the database
def update_process_state(process_id, updates=None, **kwargs):
    state = app.config["PROCESSING_STATES"].setdefault(process_id, {})
    if updates:
        state.update(updates)
    state.update(kwargs)
    process = Process.query.get(process_id)
    if process:
        combined = updates.copy() if updates else {}
        combined.update(kwargs)
        for key, value in combined.items():
            if hasattr(process, key):
                setattr(process, key, value)
        if "end_time" in combined and process.start_time and process.end_time:
            process.duration = (process.end_time - process.start_time).total_seconds()
        db.session.commit()


# Index page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "wapco" and password == "wapco":
            session["logged_in"] = True
            session["username"] = username
            flash("ورود با موفقیت انجام شد.")
            return redirect(url_for("file_selection"))
        else:
            flash("نام کاربری یا کلمه عبور اشتباه است.")
            return redirect(request.url)
    return render_template("index.html")


@app.route("/file_selection", methods=["GET", "POST"])
def file_selection():
    if not session.get("logged_in"):
        flash("لطفاً ابتدا وارد شوید.")
        return redirect(url_for("index"))

    if request.method == "POST":
        file_type = request.form.get("file_type")
        if file_type == "video":
            return redirect(url_for("video_upload"))
        elif file_type == "zip":
            return redirect(url_for("zip_upload"))
        else:
            flash("لطفاً نوع فایل مورد نظر خود را انتخاب کنید.")
            return redirect(request.url)
    return render_template("file_selection.html")


# Clear session on logout
@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    session.pop("username", None)
    flash("شما با موفقیت خارج شدید.")
    return redirect(url_for("index"))


# New route for progress updates
@app.route("/progress/<process_id>")
def progress(process_id):
    proc = Process.query.get(process_id)
    if proc:
        progress_data = {
            "status": proc.status,
            "progress": proc.progress or 0,
            "message": proc.message or "",
            "output_foldername": proc.output_folder,
        }
    else:
        progress_data = {
            "status": "not_found",
            "progress": 0,
            "message": "Process not found",
        }
    return json.dumps(progress_data)


# Video upload route
@app.route("/video-upload", methods=["GET", "POST"])
def video_upload():
    if not session.get("logged_in"):
        flash("لطفاً ابتدا وارد شوید.")
        return redirect(url_for("index"))

    if request.method == "POST":
        if "video" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["video"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename, app.config["ALLOWED_VIDEO_EXTENSIONS"]):
            filename = secure_filename(file.filename)
            process_uuid = str(uuid.uuid4())
            upload_process_dir = os.path.join(app.config["UPLOAD_FOLDER"], process_uuid)
            os.makedirs(upload_process_dir, exist_ok=True)
            video_path = os.path.join(upload_process_dir, filename)
            file.save(video_path)

            process_id = str(uuid.uuid4())
            output_dir = os.path.join(app.config["OUTPUT_FOLDER"], process_id)
            os.makedirs(output_dir, exist_ok=True)

            classify_images = request.form.get("classify_images") == "on"

            db_process = Process(
                process_uuid=process_id,
                filename=filename,
                user=session.get("username", "unknown"),
                frame_count=0,
                start_time=datetime.utcnow(),
                status="processing",
            )
            db.session.add(db_process)

            db.session.commit()

            app.config["PROCESSING_STATES"][process_id] = {
                "status": "processing",
                "progress": 0,
                "message": "در حال پردازش اولیه و آماده‌سازی...",
                "filename": filename,
                "output_foldername": process_id,
            }

            start_time_str = request.form.get("start_time", "0")
            end_time_str = request.form.get("end_time", "")
            frame_interval_str = request.form.get("frame_interval", "1")
            crop_height_ratio_str = request.form.get("crop_height_ratio", "0.1")
            model_format = request.form.get("model_format", "obj")

            try:
                start_time = float(start_time_str)
                end_time = float(end_time_str) if end_time_str else None
                frame_interval = float(frame_interval_str)
                crop_height_ratio = float(crop_height_ratio_str)
            except ValueError as e:
                update_process_state(
                    process_id,
                    {
                        "status": "failed",
                        "message": f"خطا در مقادیر ورودی زمان یا بازه: {str(e)}",
                        "end_time": datetime.utcnow(),
                    },
                )
                proc = Process.query.filter_by(process_uuid=process_id).first()
                if proc:
                    proc.status = "failed"
                    proc.end_time = datetime.utcnow()
                    if proc.start_time:
                        proc.duration = (proc.end_time - proc.start_time).total_seconds()
                    db.session.commit()
                logging.error(f"Input value error: {e}")
                return redirect(url_for("processing", process_id=process_id))

            def process_video_task(
                process_id,
                video_path,
                output_dir,
                start_time,
                end_time,
                frame_interval,
                crop_height_ratio,
                model_format,
                classify_images,
            ):
                with app.app_context():
                    try:
                        image_dir = os.path.join(output_dir, "frames")
                        os.makedirs(image_dir, exist_ok=True)

                        update_process_state(process_id, 
                            {"progress": 5, "message": "در حال استخراج فریم‌ها..."}
                        )

                        extract_command = [
                            sys.executable,
                            METASHAPE_SCRIPT_PATH,
                            "--extract_frames",
                            video_path,
                            "--image_dir",
                            image_dir,
                            "--start_time",
                            str(start_time),
                            "--frame_interval",
                            str(frame_interval),
                            "--crop_height_ratio",
                            str(crop_height_ratio),
                        ]
                        if end_time is not None:
                            extract_command.extend(["--end_time", str(end_time)])

                        logging.info(
                            f"Running frame extraction command: {' '.join(extract_command)}"
                        )
                        extract_process = subprocess.Popen(
                            extract_command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )

                        # Monitor frame extraction output - rudimentary progress
                        # You might need to modify metashape_script.py to output more detailed progress
                        total_frames = 0  # You could try to get total frames from video info before extraction
                        extracted_count = 0
                        for line in iter(extract_process.stdout.readline, b""):
                            line_str = line.decode().strip()
                            logging.info(f"ExtractFrames: {line_str}")
                            # Simple check for extracted frame count (depends on script's output format)
                            if "Extracted frame" in line_str:
                                extracted_count += 1
                                # Update progress based on extracted count if total is known
                                # if total_frames > 0:
                                #     progress_percentage = 5 + int(15 * (extracted_count / total_frames)) # 5 to 20%
                                #     update_process_state(process_id, {"progress": progress_percentage})

                        extract_process.wait()

                        if extract_process.returncode != 0:
                            stderr_output = (
                                extract_process.stderr.read().decode().strip()
                            )
                            logging.error(f"Frame extraction error: {stderr_output}")
                            update_process_state(
                                process_id,
                                {
                                    "status": "failed",
                                    "message": f"خطا در استخراج فریم‌ها: {stderr_output}",
                                    "end_time": datetime.utcnow(),
                                },
                            )
                            proc = Process.query.filter_by(process_uuid=process_id).first()
                            if proc:
                                proc.status = "failed"
                                proc.end_time = datetime.utcnow()
                                if proc.start_time:
                                    proc.duration = (proc.end_time - proc.start_time).total_seconds()
                                db.session.commit()
                            return  # Stop processing on error

                        # --- Check if any images were extracted ---
                        extracted_image_files = [
                            f
                            for f in os.listdir(image_dir)
                            if allowed_file(f, app.config["ALLOWED_IMAGE_EXTENSIONS"])
                        ]
                        if not extracted_image_files:
                            logging.error(
                                f"Frame extraction completed but no valid image files found in {image_dir}."
                            )
                            update_process_state(
                                process_id,
                                {
                                    "status": "failed",
                                    "message": "استخراج فریم‌ها انجام شد، اما هیچ فایل تصویری یافت نشد. احتمالاً ویدئو مشکل دارد یا پارامترهای استخراج نادرست هستند.",
                                    "end_time": datetime.utcnow(),
                                },
                            )
                            proc = Process.query.filter_by(process_uuid=process_id).first()
                            if proc:
                                proc.status = "failed"
                                proc.end_time = datetime.utcnow()
                                if proc.start_time:
                                    proc.duration = (proc.end_time - proc.start_time).total_seconds()
                                db.session.commit()
                            return  # Stop processing if no images were extracted

                        logging.info(
                            f"Frame extraction completed. Found {len(extracted_image_files)} images."
                        )
                        proc = Process.query.filter_by(process_uuid=process_id).first()
                        if proc:
                            proc.frame_count = len(extracted_image_files)
                            db.session.commit()
                        app.config["PROCESSING_STATES"][process_id].update(
                            {
                                "progress": 20,
                                "message": f"استخراج فریم‌ها کامل شد. یافت شد: {len(extracted_image_files)} تصویر.",
                                "frame_count": len(extracted_image_files),
                            },
                        )

                        images_to_process_dir = image_dir  # Default to original frames
                        blended_image_dir = os.path.join(
                            output_dir, "blended_images"
                        )  # Define blended output dir

                        # --- CLASSIFICATION AND BLENDING STEP ---
                        if classify_images:
                            update_process_state(process_id, 
                                {
                                    "progress": 25,
                                    "message": "در حال طبقه‌بندی و ترکیب تصاویر...",
                                }
                            )
                            try:
                                # classify_images_in_folder returns a list of paths to blended images
                                blended_image_paths = (
                                    image_classifier.classify_images_in_folder(
                                        image_dir, blended_image_dir
                                    )
                                )

                                # --- Check if any blended images were generated ---
                                if blended_image_paths:
                                    logging.info(
                                        f"Generated {len(blended_image_paths)} blended images."
                                    )
                                    # If classification and blending successful, use blended images for Metashape
                                    images_to_process_dir = blended_image_dir  # Use blended images for Metashape
                                    update_process_state(process_id, 
                                        {
                                            "progress": 40,
                                            "message": f"طبقه‌بندی و ترکیب تصاویر کامل شد. استفاده از {len(blended_image_paths)} تصویر ترکیبی برای geoSphereAi.",
                                        }
                                    )
                                else:
                                    # If classification failed or produced no blended images, log and continue with original images
                                    logging.warning(
                                        "Image classification and blending failed or produced no output blended images. Continuing with original images."
                                    )
                                    update_process_state(process_id, 
                                        {
                                            "message": "طبقه‌بندی و ترکیب تصاویر انجام نشد یا با خطا مواجه شد. ادامه با تصاویر اصلی."
                                        }
                                    )
                                    images_to_process_dir = (
                                        image_dir  # Revert to original images
                                    )

                            except Exception as e:
                                logging.error(
                                    f"Image classification and blending failed: {e}"
                                )
                                # Log error and continue with original images
                                update_process_state(process_id, 
                                    {
                                        "message": f"خطا در طبقه‌بندی و ترکیب تصاویر: {str(e)}. ادامه با تصاویر اصلی..."
                                    }
                                )
                                images_to_process_dir = (
                                    image_dir  # Revert to original images
                                )
                        else:
                            update_process_state(process_id, 
                                {
                                    "progress": 25,
                                    "message": "طبقه‌بندی تصاویر فعال نیست. شروع پردازش geoSphereAi...",
                                }
                            )
                            images_to_process_dir = image_dir  # Use original images if classification is off
                        # --- END CLASSIFICATION AND BLENDING STEP ---

                        # --- Final check before Metashape ---
                        images_for_metashape = [
                            f
                            for f in os.listdir(images_to_process_dir)
                            if allowed_file(f, app.config["ALLOWED_IMAGE_EXTENSIONS"])
                        ]
                        if not images_for_metashape:
                            logging.error(
                                f"The directory designated for geoSphereAi ({images_to_process_dir}) is empty or contains no valid images."
                            )
                            update_process_state(
                                process_id,
                                {
                                    "status": "failed",
                                    "message": "هیچ فایل تصویری معتبری برای پردازش geoSphereAi یافت نشد.",
                                    "end_time": datetime.utcnow(),
                                },
                            )
                            proc = Process.query.filter_by(process_uuid=process_id).first()
                            if proc:
                                proc.status = "failed"
                                proc.end_time = datetime.utcnow()
                                if proc.start_time:
                                    proc.duration = (proc.end_time - proc.start_time).total_seconds()
                                db.session.commit()
                            return  # Stop processing if no images for Metashape

                        logging.info(
                            f"Starting geoSphereAi process with {len(blended_image_dir)} images from {images_to_process_dir}."
                        )
                        update_process_state(process_id, 
                            {
                                "progress": 45,
                                "message": "در حال اجرای پایپ لاین geoSphereAi...",
                            }
                        )

                        metashape_command = [
                            METASHAPE_EXECUTABLE,
                            "-r",
                            METASHAPE_SCRIPT_PATH,
                            "--image_full_pipeline",
                            "--image_dir",
                            blended_image_dir,
                            "--output_dir",
                            output_dir,
                        ]

                        logging.info(
                            f"Running geoSphereAi command: {' '.join(metashape_command)}"
                        )
                        process = subprocess.Popen(
                            metashape_command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )

                        for line in iter(process.stdout.readline, b""):
                            logging.info(f"Metashape: {line.decode().strip()}")
                            line_str = line.decode().strip().lower()
                            if "aligncameras" in line_str:
                                update_process_state(process_id, 
                                    {
                                        "progress": 50,
                                        "message": "geoSphereAi: تطبیق دوربین‌ها",
                                    }
                                )
                            elif "builddepthmaps" in line_str:
                                update_process_state(process_id, 
                                    {
                                        "progress": 70,
                                        "message": "geoSphereAi: ساخت نقشه‌های عمق",
                                    }
                                )
                            elif "buildpointcloud" in line_str:
                                update_process_state(process_id, 
                                    {
                                        "progress": 85,
                                        "message": "geoSphereAi: ساخت ابر نقاط",
                                    }
                                )
                            elif "exportpointcloud" in line_str:
                                update_process_state(process_id, 
                                    {
                                        "progress": 95,
                                        "message": "geoSphereAi: خروجی ابر نقاط",
                                    }
                                )

                        process.wait()

                        if process.returncode != 0:
                            stderr_output = process.stderr.read().decode().strip()
                            logging.error(f"Metashape error: {stderr_output}")
                            update_process_state(
                                process_id,
                                {
                                    "status": "failed",
                                    "message": f"خطا در پردازش geoSphereAi: {stderr_output}",
                                    "end_time": datetime.utcnow(),
                                },
                            )
                            proc = Process.query.filter_by(process_uuid=process_id).first()
                            if proc:
                                proc.status = "failed"
                                proc.end_time = datetime.utcnow()
                                if proc.start_time:
                                    proc.duration = (proc.end_time - proc.start_time).total_seconds()
                                db.session.commit()
                            return

                        update_process_state(process_id,
                            {
                                "progress": 100,
                                "status": "completed",
                                "message": "پردازش با موفقیت انجام شد!",
                                "end_time": datetime.utcnow(),
                            }
                        )
                        proc = Process.query.filter_by(process_uuid=process_id).first()
                        if proc:
                            proc.status = "completed"
                            proc.end_time = datetime.utcnow()
                            if proc.start_time:
                                proc.duration = (proc.end_time - proc.start_time).total_seconds()
                            db.session.commit()
                        logging.info(f"Process {process_id} completed successfully.")

                    except Exception as e:
                        logging.error(
                            f"Unexpected error during processing for {process_id}: {str(e)}"
                        )
                        update_process_state(
                            process_id,
                            {
                                "status": "failed",
                                "message": f"خطای غیرمنتظره: {str(e)}",
                                "end_time": datetime.utcnow(),
                            },
                        )
                        proc = Process.query.filter_by(process_uuid=process_id).first()
                        if proc:
                            proc.status = "failed"
                            proc.end_time = datetime.utcnow()
                            if proc.start_time:
                                proc.duration = (proc.end_time - proc.start_time).total_seconds()
                            db.session.commit()

            Thread(
                target=process_video_task,
                args=(
                    process_id,
                    video_path,
                    output_dir,
                    start_time,
                    end_time,
                    frame_interval,
                    crop_height_ratio,
                    model_format,
                    classify_images,
                ),
            ).start()

            return redirect(url_for("processing", process_id=process_id))

    return render_template("video_upload.html")


# ZIP upload page
@app.route("/zip-upload", methods=["GET", "POST"])
def zip_upload():
    if not session.get("logged_in"):
        flash("لطفاً ابتدا وارد شوید.")
        return redirect(url_for("index"))

    if request.method == "POST":
        if "zip" not in request.files:
            flash("فایل ZIP یافت نشد؛ مطمئن شوید که فرم به درستی ارسال شده است.")
            logging.debug("Debug: No 'zip' key in request.files")
            return redirect(request.url)

        zip_file = request.files["zip"]
        if zip_file.filename == "":
            flash("هیچ فایلی انتخاب نشده است.")
            logging.debug("Debug: Empty filename =" + zip_file.filename)
            return redirect(request.url)

        logging.debug("Debug: Uploaded filename = " + zip_file.filename)
        if not allowed_file(zip_file.filename, app.config["ALLOWED_ZIP_EXTENSIONS"]):
            flash("فرمت فایل معتبر نیست. لطفاً فایل ZIP انتخاب کنید.")
            logging.debug("Debug: Invalid file format")
            return redirect(request.url)

        zip_filename = secure_filename(zip_file.filename)
        process_uuid = str(uuid.uuid4())
        upload_process_dir = os.path.join(app.config["UPLOAD_FOLDER"], process_uuid)
        os.makedirs(upload_process_dir, exist_ok=True)
        zip_path = os.path.join(upload_process_dir, zip_filename)

        try:
            zip_file.save(zip_path)
            logging.debug("Debug: File saved successfully at " + zip_path)
        except Exception as e:
            logging.debug("Debug: File save error: " + str(e))
            flash("مشکلی در ذخیره‌سازی فایل ZIP رخ داد.")
            return redirect(request.url)

        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], process_uuid)
        os.makedirs(output_dir, exist_ok=True)

        classify_images = request.form.get("classify_images") == "on"

        process_id = str(uuid.uuid4())
        db_process = Process(
            process_uuid=process_id,
            filename=zip_filename,
            user=session.get("username", "unknown"),
            frame_count=0,
            start_time=datetime.utcnow(),
            status="processing",
        )
        db.session.add(db_process)

        db.session.commit()

        app.config["PROCESSING_STATES"][process_id] = {
            "status": "processing",
            "progress": 0,
            "message": "در حال استخراج تصاویر از فایل ZIP...",
            "filename": zip_filename,
            "output_foldername": process_uuid,
        }

        def process_zip_task(process_id, zip_path, output_dir, classify_images):
            with app.app_context():
                try:
                    image_dir = os.path.join(output_dir, "extracted_images")
                    os.makedirs(image_dir, exist_ok=True)

                    extracted_files_count = extract_images_from_zip(zip_path, image_dir)

                    # --- Check if any images were extracted from ZIP ---
                    if extracted_files_count == 0:
                        flash("هیچ فایل تصویری مجاز در فایل ZIP یافت نشد.")
                        logging.debug("Debug: No valid images found in ZIP")
                        update_process_state(
                            process_id,
                            {
                                "status": "failed",
                                "message": "هیچ فایل تصویری معتبری در ZIP یافت نشد.",
                                "end_time": datetime.utcnow(),
                            },
                        )
                        proc = Process.query.filter_by(process_uuid=process_id).first()
                        if proc:
                            proc.status = "failed"
                            proc.end_time = datetime.utcnow()
                            if proc.start_time:
                                proc.duration = (proc.end_time - proc.start_time).total_seconds()
                            db.session.commit()
                        # Clean up the uploaded zip file and empty output directory if no images were found
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        # Only remove output_dir if it's empty to avoid deleting files from a previous run with the same UUID (unlikely but safe)
                        if os.path.exists(output_dir) and not os.listdir(output_dir):
                            os.rmdir(output_dir)
                        return  # Stop processing

                    logging.info(f"Extracted {extracted_files_count} images from ZIP.")
                    proc = Process.query.filter_by(process_uuid=process_id).first()
                    if proc:
                        proc.frame_count = extracted_files_count
                        db.session.commit()
                    app.config["PROCESSING_STATES"][process_id].update(
                        {
                            "progress": 20,
                            "message": f"تصاویر از ZIP استخراج شدند. یافت شد: {extracted_files_count} تصویر.",
                        }
                    )

                    images_to_process_dir = image_dir  # Default to extracted images
                    blended_image_dir = os.path.join(
                        output_dir, "blended_images"
                    )  # Define blended output dir

                    # --- CLASSIFICATION AND BLENDING STEP ---
                    if classify_images:
                        update_process_state(process_id, 
                            {
                                "progress": 25,
                                "message": "در حال طبقه‌بندی و ترکیب تصاویر...",
                            }
                        )
                        try:
                            blended_image_paths = (
                                image_classifier.classify_images_in_folder(
                                    image_dir, blended_image_dir
                                )
                            )

                            # --- Check if any blended images were generated ---
                            if blended_image_paths:
                                logging.info(
                                    f"Generated {len(blended_image_paths)} blended images."
                                )
                                images_to_process_dir = blended_image_dir  # Use blended images for Metashape
                                update_process_state(process_id, 
                                    {
                                        "progress": 40,
                                        "message": f"طبقه‌بندی و ترکیب تصاویر کامل شد. استفاده از {len(blended_image_paths)} تصویر ترکیبی برای geoSphereAi.",
                                    }
                                )
                            else:
                                logging.warning(
                                    "Image classification and blending failed or produced no output blended images. Continuing with original images."
                                )
                                update_process_state(process_id, 
                                    {
                                        "message": "طبقه‌بندی و ترکیب تصاویر انجام نشد یا با خطا مواجه شد. ادامه با تصاویر اصلی."
                                    }
                                )
                                images_to_process_dir = (
                                    image_dir  # Revert to original images
                                )

                        except Exception as e:
                            logging.error(
                                f"Image classification and blending failed: {e}"
                            )
                            update_process_state(process_id, 
                                {
                                    "message": f"خطا در طبقه‌بندی و ترکیب تصاویر: {str(e)}. ادامه با تصاویر اصلی..."
                                }
                            )
                            images_to_process_dir = (
                                image_dir  # Revert to original images
                            )
                    else:
                        update_process_state(process_id, 
                            {
                                "progress": 25,
                                "message": "طبقه‌بندی تصاویر فعال نیست. شروع پردازش geoSphereAi...",
                            }
                        )
                        images_to_process_dir = (
                            image_dir  # Use original images if classification is off
                        )
                    # --- END CLASSIFICATION AND BLENDING STEP ---

                    # --- Final check before Metashape ---
                    images_for_metashape = [
                        f
                        for f in os.listdir(blended_image_dir)
                        if allowed_file(f, app.config["ALLOWED_IMAGE_EXTENSIONS"])
                    ]
                    if not images_for_metashape:
                        logging.error(
                            f"The directory designated for geoSphereAi ({blended_image_dir}) is empty or contains no valid images."
                        )
                        update_process_state(
                            process_id,
                            {
                                "status": "failed",
                                "message": "هیچ فایل تصویری معتبری برای پردازش geoSphereAi یافت نشد.",
                                "end_time": datetime.utcnow(),
                            },
                        )
                        return  # Stop processing if no images for Metashape

                    logging.info(
                        f"Starting geoSphereAi process with {len(images_for_metashape)} images from {blended_image_dir}."
                    )
                    update_process_state(process_id, 
                        {
                            "progress": 45,
                            "message": "در حال اجرای پایپ لاین geoSphereAi...",
                        }
                    )

                    metashape_command = [
                        METASHAPE_EXECUTABLE,
                        "-r",
                        METASHAPE_SCRIPT_PATH,
                        "--image_full_pipeline",
                        "--image_dir",
                        blended_image_dir,
                        "--output_dir",
                        output_dir,
                    ]

                    logging.info(
                        f"Running Metashape command: {' '.join(metashape_command)}"
                    )
                    process = subprocess.Popen(
                        metashape_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                    for line in iter(process.stdout.readline, b""):
                        logging.info(f"Metashape: {line.decode().strip()}")
                        line_str = line.decode().strip().lower()
                        if "aligncameras" in line_str:
                            update_process_state(process_id, 
                                {
                                    "progress": 50,
                                    "message": "geoSphereAi: تطبیق دوربین‌ها",
                                }
                            )
                        elif "builddepthmaps" in line_str:
                            update_process_state(process_id, 
                                {
                                    "progress": 70,
                                    "message": "geoSphereAi: ساخت نقشه‌های عمق",
                                }
                            )
                        elif "buildpointcloud" in line_str:
                            update_process_state(process_id, 
                                {
                                    "progress": 85,
                                    "message": "geoSphereAi: ساخت ابر نقاط",
                                }
                            )
                        elif "exportpointcloud" in line_str:
                            update_process_state(process_id, 
                                {
                                    "progress": 95,
                                    "message": "geoSphereAi: خروجی ابر نقاط",
                                }
                            )

                    process.wait()

                    if process.returncode != 0:
                        stderr_output = process.stderr.read().decode().strip()
                        logging.error(f"Metashape error: {stderr_output}")
                        update_process_state(
                            process_id,
                            {
                                "status": "failed",
                                "message": f"خطا در پردازش geoSphereAi: {stderr_output}",
                                "end_time": datetime.utcnow(),
                            },
                        )
                        proc = Process.query.filter_by(process_uuid=process_id).first()
                        if proc:
                            proc.status = "failed"
                            proc.end_time = datetime.utcnow()
                            if proc.start_time:
                                proc.duration = (proc.end_time - proc.start_time).total_seconds()
                            db.session.commit()
                        return

                    update_process_state(process_id,
                        {
                            "progress": 100,
                            "status": "completed",
                            "message": "پردازش با موفقیت انجام شد!",
                            "end_time": datetime.utcnow(),
                        }
                    )
                    proc = Process.query.filter_by(process_uuid=process_id).first()
                    if proc:
                        proc.status = "completed"
                        proc.end_time = datetime.utcnow()
                        if proc.start_time:
                            proc.duration = (proc.end_time - proc.start_time).total_seconds()
                        db.session.commit()
                    logging.info(f"Process {process_id} completed successfully.")

                except Exception as e:
                    logging.error(
                        f"Unexpected error during processing for {process_id}: {str(e)}"
                    )
                    update_process_state(process_id,
                        {
                            "status": "failed",
                            "message": f"خطای غیرمنتظره: {str(e)}",
                            "end_time": datetime.utcnow(),
                        }
                    )
                    proc = Process.query.filter_by(process_uuid=process_id).first()
                    if proc:
                        proc.status = "failed"
                        proc.end_time = datetime.utcnow()
                        if proc.start_time:
                            proc.duration = (proc.end_time - proc.start_time).total_seconds()
                        db.session.commit()
                finally:
                    if os.path.exists(zip_path):
                        os.remove(zip_path)

        Thread(
            target=process_zip_task,
            args=(process_id, zip_path, output_dir, classify_images),
        ).start()

        return redirect(url_for("processing", process_id=process_id))

    return render_template("zip_upload.html")


# New processing page route
@app.route("/processing/<process_id>")
def processing(process_id):
    if not session.get("logged_in"):
        flash("لطفاً ابتدا وارد شوید.")
        return redirect(url_for("index"))

    if not Process.query.get(process_id):
        flash("شناسه پردازش نامعتبر است.")
        return redirect(url_for("file_selection"))

    return render_template("processing.html", process_id=process_id)


# Processes list route
@app.route("/processes")
def process_list():
    if not session.get("logged_in"):
        flash("لطفاً ابتدا وارد شوید.")
        return redirect(url_for("index"))

    page = request.args.get("page", 1, type=int)
    pagination = Process.query.order_by(Process.start_time.desc()).paginate(page=page, per_page=10)
    return render_template("process_list.html", processes=pagination.items, pagination=pagination)


# Results page route
@app.route("/results/<output_foldername>")
def results(output_foldername):
    if not session.get("logged_in"):
        flash("لطفاً ابتدا وارد شوید.")
        return redirect(url_for("index"))

    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], output_foldername)

    logging.info(f"Results route accessed for output_foldername: {output_foldername}")
    logging.info(f"Checking existence of output directory: {output_dir}")

    if not os.path.exists(output_dir):
        logging.error(f"Output directory not found for results: {output_dir}")
        flash("Results not found.")
        return redirect(url_for("file_selection"))

    logging.info(f"Output directory found: {output_dir}")
    try:
        dir_contents = os.listdir(output_dir)
        logging.info(f"Contents of output directory: {dir_contents}")
    except Exception as e:
        logging.error(f"Error listing contents of {output_dir}: {e}")

    file_paths = []
    for subdir, _, files in os.walk(output_dir):
        for file in files:
            full_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(full_path, output_dir)

            if relative_path.lower().endswith(
                (
                    ".pcd",
                    ".ply",
                    "_mask.png",
                    "_colored_mask.png",
                    "_blended.png",
                    ".obj",
                )
            ):  # Include .obj as a viewable/downloadable file
                normalized_path = relative_path.replace("\\", "/")
                file_paths.append(normalized_path)

    logging.info(f"Found {len(file_paths)} relevant files in results directory.")

    original_filename = "Processed Files"
    process = Process.query.filter_by(output_folder=output_foldername).first()
    if process:
        original_filename = process.filename or original_filename
        if process.status not in ["completed", "completed_with_warnings"]:
            logging.warning(
                f"Accessing results for process {output_foldername} which has status: {process.status}"
            )
    else:
        logging.warning(
            f"Process state not found in database for output_foldername: {output_foldername}. Using default filename."
        )

    return render_template(
        "results.html",
        filename=original_filename,
        output_foldername=output_foldername,
        file_paths=file_paths,
    )


# Download route
@app.route("/download/<output_foldername>/<path:file_path>")
def download(output_foldername, file_path):
    if not session.get("logged_in"):
        flash("لطفاً ابتدا وارد شوید.")
        return redirect(url_for("index"))

    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], output_foldername)
    full_file_path = os.path.join(output_dir, file_path)

    if not os.path.abspath(full_file_path).startswith(os.path.abspath(output_dir)):
        flash("Attempted to access a file outside the results directory.")
        logging.warning(
            f"Attempted directory traversal: {output_foldername}/{file_path}"
        )
        return redirect(url_for("results", output_foldername=output_foldername))

    if os.path.exists(full_file_path) and os.path.isfile(full_file_path):
        directory = os.path.dirname(full_file_path)
        file_name = os.path.basename(full_file_path)
        return send_from_directory(directory, file_name, as_attachment=True)
    else:
        logging.debug(f"File not found for download: {full_file_path}")
        flash("The requested file was not found on the server.")
        return redirect(url_for("results", output_foldername=output_foldername))


# New route for displaying PLY files
@app.route("/ply/<output_foldername>/<path:file_path>")
def ply(output_foldername, file_path):
    if not session.get("logged_in"):
        flash("لطفاً ابتدا وارد شوید.")
        return redirect(url_for("index"))

    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], output_foldername)
    full_file_path = os.path.join(output_dir, file_path)

    if not os.path.abspath(full_file_path).startswith(os.path.abspath(output_dir)):
        flash("Attempted to access a file outside the results directory.")
        logging.warning(
            f"Attempted directory traversal: {output_foldername}/{file_path}"
        )
        return redirect(url_for("results", output_foldername=output_foldername))

    if os.path.exists(full_file_path) and os.path.isfile(full_file_path):
        return render_template(
            "ply.html", output_foldername=output_foldername, file_path=file_path
        )
    else:
        flash("PLY file not found.")
        return redirect(url_for("results", output_foldername=output_foldername))


# New route for displaying PCD files
@app.route("/pcd/<output_foldername>/<path:file_path>")
def pcd_viewer(output_foldername, file_path):
    if not session.get("logged_in"):
        flash("لطفاً ابتدا وارد شوید.")
        return redirect(url_for("index"))

    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], output_foldername)
    full_file_path = os.path.join(output_dir, file_path)

    if not os.path.abspath(full_file_path).startswith(os.path.abspath(output_dir)):
        flash("Attempted to access a file outside the results directory.")
        logging.warning(
            f"Attempted directory traversal: {output_foldername}/{file_path}"
        )
        return redirect(url_for("results", output_foldername=output_foldername))

    if os.path.exists(full_file_path) and os.path.isfile(full_file_path):
        return render_template(
            "pcd_viewer.html", output_foldername=output_foldername, file_path=file_path
        )
    else:
        flash("pcd file not found.")
        return redirect(url_for("results", output_foldername=output_foldername))


# Static route for serving files within the output folder (e.g., for viewers)
@app.route("/outputs/<output_foldername>/<path:file_path>")
def serve_output_file(output_foldername, file_path):
    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], output_foldername)
    full_file_path = os.path.join(output_dir, file_path)

    if not os.path.abspath(full_file_path).startswith(os.path.abspath(output_dir)):
        logging.warning(
            f"Attempted directory traversal via serve_output_file: {output_foldername}/{file_path}"
        )
        return "Unauthorized", 401

    if os.path.exists(full_file_path) and os.path.isfile(full_file_path):
        directory = os.path.dirname(full_file_path)
        file_name = os.path.basename(full_file_path)
        # Determine mimetype based on file extension
        if file_name.lower().endswith(".ply"):
            mimetype = "model/ply"
        elif file_name.lower().endswith(".pcd"):
            mimetype = "text/plain"
        elif file_name.lower().endswith(".png"):
            mimetype = "image/png"
        elif file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg"):
            mimetype = "image/jpeg"
        else:
            mimetype = "application/octet-stream"

        return send_from_directory(directory, file_name, mimetype=mimetype)
    else:
        logging.debug(f"File not found for serving: {full_file_path}")
        return "File not found", 404


# Static routes for serving threejs and other static files
@app.route("/static/threejs/build/<path:filename>")
def serve_threejs_build(filename):
    threejs_build_dir = os.path.join(app.root_path, "static", "threejs", "build")
    return send_from_directory(
        threejs_build_dir, filename, mimetype="application/javascript"
    )


@app.route("/static/threejs/jsm/<path:filename>")
def serve_threejs_jsm(filename):
    threejs_jsm_dir = os.path.join(app.root_path, "static", "threejs", "jsm")
    return send_from_directory(
        threejs_jsm_dir, filename, mimetype="application/javascript"
    )


@app.route("/static/<path:filename>")
def serve_static(filename):
    static_dir = os.path.join(app.root_path, "static")
    return send_from_directory(static_dir, filename)


# Run Flask app
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False)
