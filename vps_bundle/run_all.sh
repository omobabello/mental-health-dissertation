#!/bin/bash
# Full remaining queue (resumable: finished runs are skipped). Same protocol as DistilBERT (micro-batch 8 x accumulation 2).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=$(nproc)
for job in "D1 MentalBERT 42" "D2 MentalBERT 42" "D1 MentalBERT 1337" "D2 MentalBERT 1337" "D1 MentalBERT 2026" "D2 MentalBERT 2026" \
  "D1 DistilBERT 42 --n_train 250" "D1 DistilBERT 42 --n_train 500" "D1 DistilBERT 42 --n_train 1000" "D1 DistilBERT 42 --n_train 2000" "D1 DistilBERT 42 --n_train 4000" \
  "D2 DistilBERT 42 --n_train 250" "D2 DistilBERT 42 --n_train 500" "D2 DistilBERT 42 --n_train 1000" "D2 DistilBERT 42 --n_train 2000" "D2 DistilBERT 42 --n_train 4000"; do
  tag="${job// /_}"
  echo "[$(date '+%F %T')] START $job"
  .venv/bin/python 03_transformers.py $job > "logs/$tag.log" 2>&1
  echo "[$(date '+%F %T')] END $job exit=$? $(grep DONE logs/$tag.log)"
done
echo "[$(date '+%F %T')] ALL DONE"
tar czf results_vps.tar.gz results/transformers logs run_all.log 2>/dev/null
echo "Results packed: $(pwd)/results_vps.tar.gz"
