#!/bin/bash
module load Mambaforge/23.11.0-fasrc01
module load cuda/12.4.1 cudnn/9.10.2.21_cuda12

if [ -f "/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh" ]; then
    source "/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh"
fi

ENV_CREATED=0
if mamba env list | grep -q "^hermes "; then
    echo "✓ HERMES environment already exists, skipping creation"
else
    echo "Creating HERMES environment..."
    mamba create --name hermes python=3.11 pip -y
    ENV_CREATED=1
fi

mamba activate hermes

# Install PyTorch via conda/mamba first (more reliable on HPC systems)
# Check if PyTorch imports correctly with _C module
if ! python -c "import torch; getattr(torch, '_C')" &>/dev/null; then
    echo "Installing PyTorch via conda (PyTorch installation appears broken)..."
    # Uninstall broken pip installations first
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    # Install CPU-only version (change to pytorch-cuda=11.8 or 12.1 if GPU is needed)
    mamba install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 cpuonly -c pytorch -y
else
    echo "✓ PyTorch already installed correctly"
fi

# Only install requirements if environment was just created or if packages aren't installed
if [ "$ENV_CREATED" -eq 1 ] || ! pip show transformers &>/dev/null; then
    echo "Installing requirements..."
    # Exclude torch packages from pip install since we install via conda
    pip install -r requirements.txt --ignore-installed torch torchvision torchaudio
else
    echo "✓ Requirements already installed, skipping pip install"
fi

python --version