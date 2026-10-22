Dissertation transformer runs (CPU VPS). Data = de-identified train/val/test splits (no author field).
1) bash setup.sh                  (one time, ~5 min)
2) .venv/bin/hf auth login        (paste your HF token; stays on this server)
3) bash bench.sh                  (~10 min speed test; note the two result lines)
4) tmux new -s diss   then:   bash run_all.sh 2>&1 | tee run_all.log
   (detach with Ctrl+B then D; reattach later with: tmux attach -t diss)
5) When it prints ALL DONE, copy results_vps.tar.gz back to the Mac.
6) Afterwards delete the data:  rm -rf ~/dissertation  (ethics: no retention)
