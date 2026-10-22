#!/bin/bash
# One-time setup on Ubuntu 24.04 (CPU-only). Run from inside the unpacked folder.
set -e
cd "$(dirname "$0")"
apt-get update -y && apt-get install -y python3-venv python3-pip tmux
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-cpu.txt
.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| threads', torch.get_num_threads())"
echo
echo "Setup done. Next: log in to Hugging Face (needed for MentalBERT):"
echo "  .venv/bin/hf auth login"
