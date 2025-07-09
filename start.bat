@echo off
REM — ensure drive is correct
cd /d D:\AI\3dRecognition\geoSphereAi_v1.0.1

REM — activate conda environment
CALL conda activate geosphereai

REM — run your app
python app.py

REM — pause so the window stays open
pause
