# piscVI Setup Guide

## Overview

This guide provides instructions for setting up the environment to run the pathway-informed scVI (piscVI) model.

## Requirements Summary

### Core Dependencies

- **Python**: 3.9 - 3.11 (recommended: 3.10)
- **PyTorch**: ≥2.0.0 (with CUDA support for GPU)
- **scvi-tools**: ≥1.0.0 (core framework)
- **scanpy**: ≥1.9.0 (single-cell analysis)
- **anndata**: ≥0.8.0 (data structure)
- **PyTorch Lightning**: ≥2.0.0 (training framework)
- **scikit-learn**: ≥1.0.0 (machine learning utilities)
- **scib-metrics**: ≥0.5.0 (benchmarking)

### Additional Dependencies

- numpy, pandas, scipy (scientific computing)
- matplotlib, seaborn, plottable (visualization)
- pynndescent (nearest neighbor search)
- tqdm (progress bars)

## Installation Methods

### Method 1: Using pixi (Recommended for Conda users)

```bash
# Navigate to the piscvi directory
cd /cluster/work/bewi/members/rquiles/piscvi

# Install dependencies using pixi
pixi install

# Activate the environment
pixi shell

# For GPU support, edit pixi.toml to enable CUDA feature
# Then run: pixi install --feature cuda
```

### Method 2: Using pip + virtualenv

```bash
# Create a virtual environment
python -m venv piscvi_env

# Activate the environment
source piscvi_env/bin/activate  # On Linux/Mac
# or
piscvi_env\Scripts\activate  # On Windows

# Install PyTorch with CUDA support (if GPU available)
# Visit https://pytorch.org/get-started/locally/ for the right command
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Method 3: Using Conda

```bash
# Create a new conda environment
conda create -n piscvi python=3.10

# Activate the environment
conda activate piscvi

# Install PyTorch with CUDA (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Or install via conda where possible:
conda install -c conda-forge -c bioconda numpy pandas scipy matplotlib seaborn scikit-learn tqdm
conda install -c bioconda scanpy anndata
pip install scvi-tools scib-metrics lightning plottable pynndescent
```

## Verifying Installation

After installation, verify that key packages are available:

```python
import torch
import scvi
import scanpy as sc
import lightning
import anndata
import scib_metrics

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"scvi-tools version: {scvi.__version__}")
print(f"scanpy version: {sc.__version__}")
print(f"Lightning version: {lightning.__version__}")
```

## Project Structure

```
piscvi/
├── src/
│   ├── model.py              # InformedSCVI model definition
│   ├── vae.py                # InformedVAE module
│   ├── train.py              # Training utilities
│   ├── pathway.py            # Pathway mask utilities
│   ├── utils.py              # General utilities
│   ├── run_model.py          # Main execution script
│   ├── scib_core_gmm.py      # Benchmarking with GMM clustering
│   └── scib_nmi_ari_gmm.py   # GMM-based metrics
├── data/                     # Data directory
├── results/                  # Output directory
├── notebooks/                # Jupyter notebooks
├── requirements.txt          # Pip requirements
├── pixi.toml                # Pixi configuration
└── README_SETUP.md          # This file
```

## Running the Code

### Basic Usage

```bash
# Run the main model
python src/run_model.py

# Run training
python src/train.py

# Run benchmarking
python src/benchmark.py
```

### Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ directory and open desired notebook
```

## GPU Support

### Checking GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
```

### Installing CUDA-enabled PyTorch

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to get the correct installation command for your CUDA version.

Common examples:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Common Issues and Solutions

### Issue: ImportError for scvi modules

**Solution**: Ensure scvi-tools is installed correctly
```bash
pip install --upgrade scvi-tools
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or use gradient accumulation
- Edit training parameters in your script
- Use smaller model dimensions

### Issue: Module not found errors

**Solution**: Ensure you're in the correct directory
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/piscvi/src"

# Or run from project root:
cd /cluster/work/bewi/members/rquiles/piscvi
python -m src.run_model
```

### Issue: Lightning compatibility errors

**Solution**: Ensure compatible versions
```bash
pip install lightning>=2.0.0 scvi-tools>=1.0.0
```

## Development Setup

For development, install additional tools:

```bash
pip install black pytest jupyter ipykernel

# Setup Jupyter kernel
python -m ipykernel install --user --name=piscvi --display-name="Python (piscvi)"
```

## Additional Resources

- **scvi-tools documentation**: https://docs.scvi-tools.org/
- **scanpy documentation**: https://scanpy.readthedocs.io/
- **PyTorch documentation**: https://pytorch.org/docs/
- **Lightning documentation**: https://lightning.ai/docs/pytorch/

## Contact

For issues specific to this implementation, refer to the project repository or contact the maintainers.
