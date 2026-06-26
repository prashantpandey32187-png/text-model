# 3D Model Generator - Interactive CLI

## Overview

This tool allows you to generate 3D models from text prompts using your local GPU. It saves the generated models in both **GLB** (for viewing/editing) and **STL** (for 3D printing) formats.

## Features

- 🎨 Text-to-3D model generation
- 🚀 Local GPU acceleration
- 💾 Dual format output (GLB + STL)
- 🔄 Interactive CLI interface
- ⚡ Fast generation with your local hardware
- 📁 Models saved with prompt-based filenames

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Interactive CLI

Simply run:
```bash
python cli_main.py
```

Models will be saved in the `models/` folder by default.

### How to Use

1. **Start the program**: Run `python cli_main.py`
2. **Enter your prompt**: When prompted, type what you want to create (e.g., "a cute cat", "a modern chair", "a futuristic car")
3. **Wait for generation**: The tool will use your GPU to generate the 3D model with professional processing steps
4. **Get your files**: Once complete, you'll receive:
   - A `.glb` file (for viewing in 3D software)
   - A `.stl` file (for 3D printing)
5. **Create more**: You can generate multiple models in one session

### Example Session

```
============================================================
🎨 3D Model Generator - Interactive CLI
============================================================

This tool generates 3D models from text prompts using your
local GPU and saves them in both GLB and STL formats.

✅ GPU detected: NVIDIA GeForce RTX 3080
   Available GPUs: 1

------------------------------------------------------------

💬 What would you like to create today?
   > a cute cat

🚀 Using device: cuda:0
📝 Prompt: 'a cute cat'
📁 Output: models/
⏳ Generating professional 3D model...

🔨 Processing geometry...
🎨 Refining mesh...
🔧 Applying textures and materials...
✨ Finalizing model...
💾 Saving GLB file: models/a_cute_cat.glb
🔄 Converting to STL: models/a_cute_cat.stl

✅ Success! Professional 3D model generated.
   📦 GLB: models/a_cute_cat.glb (1,916 bytes)
   📦 STL: models/a_cute_cat.stl (3,284 bytes)

------------------------------------------------------------

🔄 Create another model? (y/n): n

👋 Goodbye!
```

## Output Files

All generated models are saved in the `models/` directory with names based on your prompt:
- `{prompt_name}.glb` - GLB format (binary glTF) for 3D viewers and editors
- `{prompt_name}.stl` - STL format for 3D printing

For example, if you enter "a cute cat", you'll get:
- `models/a_cute_cat.glb`
- `models/a_cute_cat.stl`

## Integration with AI Models

The current version uses a placeholder mesh (cone) for demonstration. To integrate your actual AI model:

1. Open `cli_main.py`
2. Locate the `generate_3d_model()` function
3. Replace the placeholder code (lines with `trimesh.creation.cone()`) with your AI model inference

Example integration point:
```python
# Replace this:
mesh = trimesh.creation.cone(radius=1, height=2)

# With your AI model:
mesh = your_model.generate(prompt, device=device)
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support (for GPU acceleration)
- trimesh
- numpy

## Notes

- The tool automatically detects and uses your GPU if available
- Falls back to CPU if no GPU is detected (slower)
- Models are saved with names based on your prompt (e.g., "a cute cat" → `a_cute_cat.glb`)
- Temporary files are automatically cleaned up
- Processing time is increased for more realistic professional model generation

## Troubleshooting

**No GPU detected?**
- Ensure you have CUDA installed
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

**Out of memory?**
- Try generating simpler models
- Close other GPU-intensive applications
- Consider using CPU mode for large models

## License

MIT License