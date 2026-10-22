#!/bin/zsh
# Sequential, resumable transformer queue (each run skips if its JSON exists).
# Usage: ./run_queue.sh "D1 DistilBERT 42" "D2 DistilBERT 42" ...
cd "$(dirname "$0")"
mkdir -p results/transformers logs
for job in "$@"; do
  tag="${job// /_}"
  echo "[$(date +%H:%M:%S)] START $job"
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python 03_transformers.py ${=job} > "logs/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] END $job exit=$? $(grep DONE logs/$tag.log)"
done
