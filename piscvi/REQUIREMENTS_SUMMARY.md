# piscVI Requirements Summary

## Quick Reference - Core Dependencies

### Essential Packages

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.9 - 3.11 | Programming language |
| **torch** | ≥2.0.0 | Deep learning framework |
| **scvi-tools** | ≥1.0.0 | Single-cell variational inference toolkit (core framework) |
| **scanpy** | ≥1.9.0 | Single-cell analysis toolkit |
| **anndata** | ≥0.8.0 | Annotated data structure for single-cell data |
| **lightning** | ≥2.0.0 | PyTorch Lightning (training framework) |

### Scientific Computing

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | ≥1.21.0 | Numerical computing |
| **pandas** | ≥1.3.0 | Data manipulation |
| **scipy** | ≥1.7.0 | Scientific computing |

### Machine Learning & Benchmarking

| Package | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | ≥1.0.0 | Machine learning utilities |
| **scib-metrics** | ≥0.5.0 | Single-cell integration benchmarking |
| **pynndescent** | ≥0.5.8 | Fast approximate nearest neighbors |

### Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | ≥3.5.0 | Basic plotting |
| **seaborn** | ≥0.11.0 | Statistical visualizations |
| **plottable** | ≥0.1.5 | Table visualizations (for benchmarking) |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| **tqdm** | ≥4.62.0 | Progress bars |

## Installation Files Provided

1. **requirements.txt** - For pip installation
2. **pixi.toml** - For pixi/conda installation
3. **setup.sh** - Automated setup script
4. **README_SETUP.md** - Detailed setup guide

## Quick Start Commands

### Using pip:
```bash
python -m venv piscvi_env
source piscvi_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version
pip install -r requirements.txt
```

### Using conda:
```bash
conda create -n piscvi python=3.10
conda activate piscvi
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Using pixi:
```bash
pixi install
pixi shell
```

### Using the setup script:
```bash
bash setup.sh cuda118  # Options: cpu, cuda118, cuda121
```

## Key Dependencies Explained

### scvi-tools
The core framework that provides:
- VAE architecture components (Encoder, Decoder, FCLayers)
- Training infrastructure (TrainingPlan, TrainRunner)
- Data management (AnnDataManager, DataSplitter)
- Registry and field systems for data handling

This package will automatically install many of the required dependencies.

### PyTorch Lightning (lightning)
Provides:
- LightningDataModule for data handling
- Simplified training loops
- Multi-GPU and distributed training support

### scanpy
Used for:
- Single-cell data preprocessing
- Quality control
- Visualization
- Data I/O

### Custom Modules
The project includes custom implementations in `src/`:
- `scib_core_gmm.py` - Modified benchmarking with GMM clustering
- `scib_nmi_ari_gmm.py` - GMM-based NMI/ARI metrics
These extend the standard scib-metrics package with Gaussian Mixture Model clustering.

## Platform Requirements

- **OS**: Linux (tested), macOS (should work), Windows (may need adjustments)
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA support)
- **RAM**: Minimum 16GB recommended for typical datasets
- **Storage**: Depends on dataset size; recommend 10GB+ free space

## Python Version Compatibility

- **Python 3.9**: Supported ✓
- **Python 3.10**: Recommended ✓
- **Python 3.11**: Supported ✓
- **Python 3.12**: May have compatibility issues with some dependencies

## GPU/CUDA Setup

### Checking CUDA Compatibility
```bash
nvidia-smi  # Check NVIDIA driver and CUDA version
```

### PyTorch CUDA Installation
Visit https://pytorch.org/get-started/locally/ to get the exact command for your setup.

Common configurations:
- **CUDA 11.8**: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **CUDA 12.1**: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- **CPU only**: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Verification

After installation, verify with:
```python
import torch, scvi, scanpy as sc, lightning, anndata
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"scvi-tools: {scvi.__version__}, scanpy: {sc.__version__}")
```

## Troubleshooting

### Common Issues:
1. **CUDA version mismatch**: Ensure PyTorch CUDA version matches your system CUDA
2. **Import errors**: Check that you're in the correct environment
3. **Memory errors**: Reduce batch size or use smaller models
4. **Module not found**: Add `src/` to PYTHONPATH or run from project root

See `README_SETUP.md` for detailed troubleshooting steps.

## Additional Notes

- The project uses pathway information from GMT files (stored in `src/resources/`)
- Data should be in AnnData format (.h5ad files)
- Results are saved to the `results/` directory
- Benchmarking uses modified scib-metrics with GMM clustering support

## For Cluster/HPC Users

If running on a cluster:
1. Load appropriate modules (Python, CUDA)
2. Consider using a job scheduler (SLURM, PBS)
3. Set appropriate memory and GPU requirements
4. Use virtual environments or conda for isolation

Example SLURM header:
```bash
#!/bin/bash
#SBATCH --job-name=piscvi
#SBATCH --output=piscvi_%j.out
#SBATCH --error=piscvi_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

module load python/3.10
module load cuda/11.8
source piscvi_env/bin/activate
python src/run_model.py
```
