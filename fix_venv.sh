#!/bin/bash
cd /home/luoyashuo/research/graph_kd/cdad

# Deactivate current venv if active
deactivate 2>/dev/null

# Remove old venvs
echo "Removing old virtual environments..."
rm -rf .venv .venv_new

# Create new venv with Python 3.12
echo "Creating new virtual environment with Python 3.12..."
python3.12 -m venv .venv

# Activate new venv
source .venv/bin/activate

# Upgrade pip and install setuptools
echo "Upgrading pip and installing setuptools..."
pip install --upgrade pip setuptools wheel

# Install dependencies from requirements file
echo "Installing dependencies (this will take several minutes)..."
pip install -r LlamaFactory/requirements.txt

echo "Done! Activate with: source /home/luoyashuo/research/graph_kd/cdad/.venv/bin/activate"
