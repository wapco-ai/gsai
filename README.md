# GeoSphereAI

This project processes video or image archives using Agisoft Metashape. Before running `app.py` make sure the following configuration is available:

1. **Proxy settings (Windows only)**
   - The application reads the current user's Windows proxy configuration. The proxy is only applied when running on Windows.

2. **Metashape executable location**
   - Set the `METASHAPE_EXECUTABLE` environment variable to the path of the Metashape executable, for example:
     ```bash
     export METASHAPE_EXECUTABLE="C:\\Program Files\\Agisoft\\Metashape Pro\\metashape.exe"
     ```
   - Alternatively create a `config.json` file next to `app.py` with the following structure:
     ```json
     {
       "METASHAPE_EXECUTABLE": "C:/Program Files/Agisoft/Metashape Pro/metashape.exe"
     }
     ```
   - The application will raise an error if the executable path is not found via either method.

Run the application with `python app.py` once these settings have been configured.
