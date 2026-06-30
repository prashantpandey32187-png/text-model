# GPU Acceleration Setup Guide

This guide explains how to enable GPU acceleration for faster 3D model generation.

## Current System Status

- **GPU(s)**: AMD Radeon RX 6500M + AMD Radeon Graphics (integrated)
- **NVIDIA GPU**: Not detected
- **CUDA Toolkit**: v13.3 installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`
- **CUDA Runtime**: Not usable - no NVIDIA GPU detected
- **PyTorch Version**: 2.6.0+cu124 (CUDA 12.4 support bundled)
- **CUDA Available**: `False` (requires NVIDIA GPU)
- **Current Mode**: CPU-only

## System Requirements

This project requires an **NVIDIA GPU** with CUDA support for GPU-accelerated generation:

| GPU Type | Support | Performance |
|----------|---------|-------------|
| NVIDIA (CUDA) | ✅ Full | 5-10x faster |
| CPU (no GPU) | ✅ Works | Slower but functional |
| AMD GPU | ⚠️ Not supported by PyTorch CUDA | CPU-only fallback |

## Performance Comparison

| Mode | Generation Time | Quality |
|------|----------------|---------|
| CPU | ~30-60 seconds | Standard |
| GPU (NVIDIA CUDA) | ~5-10 seconds | Standard |
| GPU + cuDNN | ~3-5 seconds | Standard |

## If You Have an NVIDIA GPU

### 1. Install NVIDIA Drivers

Download and install the latest NVIDIA drivers from:
https://www.nvidia.com/drivers

Verify with:
```bash
nvidia-smi
```

### 2. Install CUDA Toolkit (if not already installed)

The CUDA Toolkit v13.3 is already installed. Verify:
```bash
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc" --version
```

### 3. Add CUDA to PATH (if needed)

```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set PATH=%CUDA_PATH%\bin;%PATH%
```

### 4. Install CUDA-enabled PyTorch (already installed)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is working:
```bash
python test_gpu.py
```

Expected output with NVIDIA GPU:
```
============================================================
GPU Detection Test
============================================================

PyTorch Version: 2.6.0+cu124
CUDA Available: True
CUDA Version: 12.4

[SUCCESS] GPU(s) Detected: 1
   GPU 0: NVIDIA GeForce RTX 3060
   Memory: 12.00 GB
```

## Current Installation Details

- **PyTorch**: 2.6.0+cu124
- **CUDA in PyTorch**: 12.4 (bundled)
- **CUDA Toolkit on System**: 13.3
- **Project Dependencies**: All installed

## Troubleshooting

### Issue: "CUDA not available" on system with AMD GPU

This is expected. AMD GPUs are not supported by PyTorch CUDA. The system will use CPU instead.

### Issue: "CUDA not available" on system with NVIDIA GPU

**Solution 1**: Verify NVIDIA drivers are installed
```bash
nvidia-smi
```

**Solution 2**: Check CUDA toolkit installation
```bash
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc" --version
```

**Solution 3**: Add CUDA to PATH
```cmd
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%
```

### Issue: "Out of memory" errors

**Solution**: Reduce batch size or model size in `generation/local_model.py`:
```python
# Reduce image resolution
height=512,  # Change to 384
width=512,   # Change to 384
```

## Using the Project (CPU Mode)

Simply run the generation:
```bash
python cli_main.py --prompt "a car model"
```

The system will automatically detect no GPU and use CPU.

## Files Modified

- `requirements.txt` - GPU-enabled dependencies (CUDA 12.4 wheels)
- `generation/local_model.py` - GPU/CPU auto-detection
- `test_gpu.py` - GPU verification script

## Next Steps

1. To use GPU: Install an NVIDIA GPU and corresponding drivers
2. Run `python test_gpu.py` to verify GPU detection
3. Run `python cli_main.py --prompt "a car model"` to test generation

## Notes

- First run will download AI models (~4GB) to `C:\Users\prash\.cache\text-to-3d-models`
- Subsequent runs will use cached models (much faster)
- GPU acceleration provides 5-10x speed improvement for AI-based generation
- Procedural generation works on CPU and doesn't benefit from GPU