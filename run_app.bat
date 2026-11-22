@echo off
REM Starts the app. First start downloads the engine (llama.cpp) and a model that fits this PC.
REM Needs only: pip install -r requirements-app.txt
python -m pcore.app %*
