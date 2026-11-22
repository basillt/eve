@echo off
REM Experiment 1 (experts): which experts of an MoE model fire for you vs. for everyone.
REM Usage:  run_experiment.bat                         (sample personal prompts)
REM         run_experiment.bat prompts\personal.jsonl  (your own prompts)
set MODEL=ibm-granite/granite-3.1-1b-a400m-instruct
set PERSONAL=%1
if "%PERSONAL%"=="" set PERSONAL=prompts\personal_sample.jsonl

python profile_experts.py --model %MODEL% --prompts %PERSONAL% --out results\personal.npz --skip-template || exit /b 1
python profile_experts.py --model %MODEL% --prompts prompts\generic.jsonl --out results\generic.npz --skip-template || exit /b 1
python analyze.py --personal results\personal.npz --generic results\generic.npz --outdir results
echo.
echo Done. Open results\hit_rate.png and results\summary.txt
