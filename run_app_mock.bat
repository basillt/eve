@echo off
REM Starts the app WITHOUT a model (canned replies) - to look at the UI.
python -m pcore.server --mock --data data_mock
