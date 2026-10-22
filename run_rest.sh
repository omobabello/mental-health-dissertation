#!/bin/zsh
# Sequential remainder of the experiment pipeline (resumable: finished runs are skipped).
cd "$(dirname "$0")"
./run_queue.sh "D1 MentalBERT 42 --micro_batch 4 --grad_ckpt" "D2 MentalBERT 42 --micro_batch 4 --grad_ckpt" "D1 MentalBERT 1337 --micro_batch 4 --grad_ckpt" "D2 MentalBERT 1337 --micro_batch 4 --grad_ckpt" \
  "D1 MentalBERT 2026 --micro_batch 4 --grad_ckpt" "D2 MentalBERT 2026 --micro_batch 4 --grad_ckpt" \
  "D1 DistilBERT 42 --n_train 250" "D1 DistilBERT 42 --n_train 500" "D1 DistilBERT 42 --n_train 1000" \
  "D1 DistilBERT 42 --n_train 2000" "D1 DistilBERT 42 --n_train 4000" \
  "D2 DistilBERT 42 --n_train 250" "D2 DistilBERT 42 --n_train 500" "D2 DistilBERT 42 --n_train 1000" \
  "D2 DistilBERT 42 --n_train 2000" "D2 DistilBERT 42 --n_train 4000"
echo "[$(date +%H:%M:%S)] training done"
sleep 30
.venv/bin/python 06_benchmark.py > logs/benchmark.log 2>&1; echo "[$(date +%H:%M:%S)] benchmark exit $?"
.venv/bin/python 04_analysis.py > logs/analysis.log 2>&1; echo "[$(date +%H:%M:%S)] analysis exit $?"
.venv/bin/python 05_figures.py > logs/figures.log 2>&1; echo "[$(date +%H:%M:%S)] figures exit $?"
echo "ALL DONE"
