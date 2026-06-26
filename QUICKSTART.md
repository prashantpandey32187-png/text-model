# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Generator

**Option A: Interactive Mode (Recommended)**
```bash
python cli_main.py --interactive
# or simply
python cli_main.py
```

**Option B: Single Prompt Mode**
```bash
python cli_main.py --prompt "a cute cat"
# or
python cli_main.py -p "a modern chair"
```

**Option C: Custom Output Directory**
```bash
python cli_main.py -p "a futuristic car" -o ./my_models
```

### 3. Find Your Models
Generated files are saved in the `outputs/` directory:
- `{uuid}.glb` - For viewing in 3D software (Blender, etc.)
- `{uuid}.stl` - For 3D printing

## 📝 Example Usage

```bash
# Start the interactive CLI
python cli_main.py

# You'll see:
# ============================================================
# 🎨 3D Model Generator - Interactive CLI
# ============================================================
#
# 💬 What would you like to create today?
#    > a cute cat
#
# 🚀 Using device: cuda:0
# 📝 Prompt: 'a cute cat'
# ⏳ Generating 3D model...
#
# ✅ Success! 3D model generated.
#    📦 GLB: outputs\abc-123.glb
#    📦 STL: outputs\abc-123.stl
```

## 🔧 Integration with Your AI Model

To use your actual AI model instead of the placeholder:

1. Open `cli_main.py`
2. Find the `generate_3d_model()` function (around line 45)
3. Replace the placeholder code:

```python
# CURRENT (placeholder):
mesh = trimesh.creation.cone(radius=1, height=2)

# REPLACE WITH YOUR MODEL:
# Example for a model that takes a prompt and returns a mesh:
mesh = your_model.generate(prompt, device=device)

# Or if your model returns vertices and faces:
vertices, faces = your_model.generate(prompt, device=device)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
```

## 🧪 Test the Setup

Run the test script to verify everything works:
```bash
python test_generation.py
```

## 📚 More Information

- See `README_CLI.md` for detailed documentation
- See `main.py` for the FastAPI server version (multi-GPU)
- Check `generation/` folder for mesh utilities

## 💡 Tips

- Use `--help` to see all options: `python cli_main.py --help`
- The tool auto-detects GPU and falls back to CPU if needed
- Each generation creates unique files (no overwrites)
- Temporary files are automatically cleaned up

## 🐛 Troubleshooting

**No GPU detected?**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Import errors?**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Permission errors on Windows?**
- Run terminal as Administrator
- Or use WSL (Windows Subsystem for Linux)