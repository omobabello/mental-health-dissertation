# Classical vs Transformer Models for Depression-Indicator Detection

Experiment code for the MSc Computer Science dissertation *A Comparative Evaluation of Classical Machine Learning and Transformer-Based Models for Early Detection of Depression Indicators in Social Media Text* (University of East London, 2026).

The pipeline compares TF-IDF + logistic regression, linear SVM and random forest against fine-tuned DistilBERT and MentalBERT under one protocol. It reports predictive performance (macro-F1, McNemar and paired-bootstrap tests against a pre-specified 0.02 threshold), computational cost (training time, latency, parameters) and interpretability (explanation stability and lexicon plausibility).

This is a research artefact for non-clinical screening research. It is not a diagnostic tool.

## Data

The Reddit Mental Health Dataset (Low et al., 2020), Zenodo record [3941387](https://zenodo.org/records/3941387), ODC-PDDL licence. Download the 2019 (`*_2019_features_tfidf_256.csv`) and 2020 (`*_post_features_tfidf_256.csv`) files for these subreddits into `data/`: depression, fitness, jokes, meditation, parenting, teaching, divorce, personalfinance, anxiety, lonely, mentalhealth and bpd.

**No Reddit text is included in this repository.** The `data/` folder, the fitted models and the archives are git-ignored. The per-token word lists of the Integrated Gradients examples were removed from `results/transformers/*.json` before publication; the attribution scores and every metric are kept. The `author` column is never loaded.

## Conditions

| Condition | Positive | Negative |
|---|---|---|
| D1 | r/depression (3,000) | 7 non-mental-health subreddits (7,000) |
| D2 | r/depression (3,000) | anxiety, lonely, mentalhealth, bpd (7,000; hard negatives) |
| D1T | 2020 temporal hold-out scored with the D1 models (1,500) | |

Each 2019 condition is split 70/15/15 (6,999 / 1,500 / 1,501), stratified, with seed 42.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login        # MentalBERT (mental/mental-bert-base-uncased) is gated on Hugging Face
```

## Reproduction order

1. Build the dataset (ingest, masking, conditions, splits):
   ```bash
   python 01_build_dataset.py
   ```
2. Train the classical models (grid search with 5-fold CV, refit, ablations, learning curves):
   ```bash
   python 02_classical.py
   ```
3. Fine-tune the transformers. There are 22 runs in total: two models × three seeds (42, 1337, 2026) × D1/D2, plus the DistilBERT learning curves. Each call does one run, and runs that have already finished are skipped:
   ```bash
   python 03_transformers.py D1 DistilBERT 42
   python 03_transformers.py D2 MentalBERT 1337
   python 03_transformers.py D1 DistilBERT 42 --n_train 1000 --no-ig
   ./run_queue.sh "D1 DistilBERT 42" "D2 DistilBERT 42"   # local resumable queue
   ```
4. Run the controlled latency benchmark on an otherwise idle machine:
   ```bash
   python 06_benchmark.py
   ```
5. Run the aggregation, significance tests, decision rule, interpretability and error analysis. This writes `results/analysis.json`:
   ```bash
   python 04_analysis.py
   ```
6. Draw the charts (written to `figures/`, or to `FIG_OUT` if set):
   ```bash
   python 05_figures.py
   ```
7. Test the significance of the LR ablations:
   ```bash
   python 07_ablation_tests.py
   ```
8. Re-measure classical explanation stability across the transformer seeds (like-for-like with the transformers):
   ```bash
   python 08_stability_seeds.py
   ```

Every number in the dissertation is read from `results/*.json`.

## Hardware and the VPS route

- **Laptop:** classical models, DistilBERT (PyTorch MPS) and all latency benchmarks ran on an Apple M1 laptop with 8 GB of memory.
- **VPS:** MentalBERT (three seeds on D1 and D2) and the DistilBERT learning-curve runs needed more memory. They ran on the author's privately rented VPS (AMD EPYC 9354P, 4 vCPU, 15 GB RAM, Ubuntu 24.04, CPU only) with the identical protocol.
- **VPS package:** `vps_bundle/` holds the scripts used there:
  - `setup.sh`
  - `bench.sh`
  - `run_all.sh`
  - `README.txt`
- **VPS data handling:** only the de-identified splits were uploaded. All project data were deleted from the server after the results were copied back. `run_all.log` is the queue log from that run.

## Layout

```
01_build_dataset.py … 08_stability_seeds.py   pipeline scripts, run in the order above
common.py            paths, seeds, conditions, MIN_DELTA_F1 = 0.02, clean_text, metrics
fig_style.py, figs/  shared chart theme and one module per chart
results/             all derived metrics (JSON), one file per transformer run in results/transformers/
logs/                per-run training logs
run_queue.sh         local resumable queue
vps_bundle/          VPS package (scripts only; splits are not published)
run_all.log          VPS queue log
requirements.txt     pinned library versions (Python 3.13)
```

## Citation

Bello, A. O. (2026). *A comparative evaluation of classical machine learning and transformer-based models for early detection of depression indicators in social media text* [Master's dissertation, University of East London].
