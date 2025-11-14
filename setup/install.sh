#!/usr/bin/env bash
set -e

# Move into the directory of this script so relative paths work
cd "$(dirname "$0")"

# Create venv in project root (one level up from setup/)
if [ ! -d "../venv" ]; then
    python3 -m venv ../venv
fi

source ../venv/bin/activate

# Upgrade pip tooling
pip install --upgrade pip wheel setuptools

# Install CPU-only PyTorch from official index
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Jupyter
pip install jupyterlab            # modern UI

# Install remaining dependencies
pip install -r requirements.txt

pip install -U pyarrow

pip install lightgbm


echo "✅ Environment setup complete."
echo "👉 To activate, run: source venv/bin/activate"

