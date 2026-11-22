@echo off
REM Research engine (own Python MoE engine with the personal expert core). Needs torch + requirements.txt.
set MODEL=ibm-granite/granite-3.1-1b-a400m-instruct
if not exist store\pcore_meta.json (
  python -m pcore.engine split --model %MODEL% --out store || exit /b 1
)
python -m pcore.server --store store --data data --night-at 03:00
