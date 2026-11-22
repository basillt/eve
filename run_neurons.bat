@echo off
REM Experiment 2 (neurons): which MLP neurons of a normal (dense) model fire for you vs. for everyone.
REM Fits a 4GB GPU. First run downloads the model (~3 GB).
REM Usage:  run_neurons.bat                         (sample personal prompts)
REM         run_neurons.bat prompts\personal.jsonl  (your own prompts)
set MODEL=Qwen/Qwen2.5-1.5B-Instruct
set PERSONAL=%1
if "%PERSONAL%"=="" set PERSONAL=prompts\personal_sample.jsonl

python profile_experts.py --model %MODEL% --unit neurons --prompts %PERSONAL% --out results_neurons\personal.npz --skip-template --max-new-tokens 48 --gpu-mem 3.2GiB || exit /b 1
python profile_experts.py --model %MODEL% --unit neurons --prompts prompts\generic.jsonl --out results_neurons\generic.npz --skip-template --max-new-tokens 48 --gpu-mem 3.2GiB || exit /b 1
python analyze.py --personal results_neurons\personal.npz --generic results_neurons\generic.npz --outdir results_neurons
echo.
echo Done. Open results_neurons\hit_rate.png and results_neurons\summary.txt
