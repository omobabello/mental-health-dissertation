#!/bin/bash
# ~10-minute speed test: 400 training posts (50 steps), 1 epoch, MentalBERT on D1.
cd "$(dirname "$0")"
start=$(date +%s)
.venv/bin/python 03_transformers.py D1 MentalBERT 7 --n_train 400 --epochs 1 --no-ig 2>&1 | grep -E "step|epoch|DONE|Error|rror"
end=$(date +%s)
.venv/bin/python - <<PY
import json, glob
d = json.load(open(glob.glob("results/transformers/D1_MentalBERT_7_n400.json")[0]))
ep = d["history"][0]["epoch_time_s"]
print(f"\nTotal wall time: {($end-$start)/60:.1f} min | epoch(50 steps + 1,500-post validation): {ep/60:.1f} min")
print("Record these two lines to size the queue.")
PY
rm -f results/transformers/D1_MentalBERT_7_n400.json
