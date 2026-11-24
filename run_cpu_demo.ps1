# Reduced-scale run that finishes in about an hour on a laptop CPU (no GPU needed).
# It uses 4,000 training images and 3-step PGD adversarial training, so the numbers are
# LOWER than the full-scale Colab/GPU results - it only demonstrates the effect.
# For the full experiment of the report, use the Colab notebook (or train.py on a GPU).
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Set-Location $PSScriptRoot

& $py -X utf8 train.py --mode standard --epochs 10 --train-subset 4000 --eval-batches 1 --monitor-steps 3 --out runs_cpu_demo
& $py -X utf8 train.py --mode pgd_at   --epochs 6  --train-subset 4000 --pgd-steps 3 --eval-batches 1 --monitor-steps 3 --out runs_cpu_demo
& $py -X utf8 evaluate.py --runs runs_cpu_demo --out results_cpu_demo --n-test 500 --sweep-batches 1 --sweep-steps 10
