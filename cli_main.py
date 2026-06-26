import os
import sys
import time
import uuid
import argparse
import torch
import trimesh

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generation.mesh_utils import convert_obj_to_stl
from generation.text_to_3d import TextTo3DGenerator

def sanitize_filename(prompt: str, max_length: int = 50) -> str:
    """
    Convert prompt to a safe filename
    """
    # Replace spaces with underscores, remove special characters
    filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
    filename = filename.replace(' ', '_').lower()
    # Limit length
    filename = filename[:max_length]
    # Remove trailing underscores
    filename = filename.strip('_')
    return filename if filename else "model"

def generate_3d_model(prompt: str, output_dir: str = "models") -> tuple:
    """
    Generates a 3D model from text prompt using AI.
    Returns paths to both GLB and STL files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from prompt
    filename = sanitize_filename(prompt)
    
    # Define output paths
    glb_path = os.path.join(output_dir, f"{filename}.glb")
    stl_path = os.path.join(output_dir, f"{filename}.stl")
    
    try:
        print(f"\n🚀 Starting 3D model generation...")
        print(f"📝 Prompt: '{prompt}'")
        print(f"📁 Output: {output_dir}/")
        print(f"⏳ This may take a minute...\n")
        
        # Initialize the AI generator
        generator = TextTo3DGenerator(backend='huggingface')
        
        # Generate the GLB model using AI
        print("🔨 Generating 3D model from your prompt...")
        generator.generate(prompt, glb_path)
        
        # Convert GLB to STL
        print(f"\n🔄 Converting to STL format...")
        temp_obj_path = os.path.join(output_dir, f"{filename}.obj")
        
        # Load and export as OBJ for conversion
        mesh = trimesh.load(glb_path)
        mesh.export(temp_obj_path)
        convert_obj_to_stl(temp_obj_path, stl_path)
        
        # Clean up temporary OBJ file
        if os.path.exists(temp_obj_path):
            os.remove(temp_obj_path)
        
        # Get file sizes
        glb_size = os.path.getsize(glb_path) if os.path.exists(glb_path) else 0
        stl_size = os.path.getsize(stl_path) if os.path.exists(stl_path) else 0
        
        print(f"\n✅ Success! 3D model generated from your prompt.")
        print(f"   📦 GLB: {glb_path} ({glb_size:,} bytes)")
        print(f"   📦 STL: {stl_path} ({stl_size:,} bytes)")
        
        return glb_path, stl_path
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        # Clean up partial files
        for path in [glb_path, stl_path]:
            if os.path.exists(path):
                os.remove(path)
        raise

def interactive_mode():
    """Interactive CLI for 3D model generation"""
    print("=" * 60)
    print("🎨 3D Model Generator - Interactive CLI")
    print("=" * 60)
    print("\nThis tool generates 3D models from text prompts using your")
    print("local GPU and saves them in both GLB and STL formats.\n")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Available GPUs: {gpu_count}\n")
    else:
        print("⚠️  No GPU detected. Running on CPU (slower).\n")
    
    while True:
        try:
            # Get user input
            print("-" * 60)
            prompt = input("\n💬 What would you like to create today?\n   > ").strip()
            
            if not prompt:
                print("⚠️  Please enter a prompt!")
                continue
            
            if prompt.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Generate the model
            glb_path, stl_path = generate_3d_model(prompt)
            
            # Ask if user wants to create another
            print("\n" + "-" * 60)
            another = input("\n🔄 Create another model? (y/n): ").strip().lower()
            if another != 'y':
                print("\n👋 Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("   Please try again.\n")

def main():
    """Main entry point with CLI argument support"""
    parser = argparse.ArgumentParser(
        description='3D Model Generator - Generate 3D models from text prompts'
    )
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        help='Text prompt for 3D model generation (e.g., "a cute cat")'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for generated models (default: models)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode (prompts for input)'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, run interactive mode
    if not args.prompt and not args.interactive:
        try:
            interactive_mode()
        except EOFError:
            print("\n\n⚠️  Interactive mode not available in this environment.")
            print("   Use --prompt or -p to specify a prompt directly.")
            print("   Example: python cli_main.py --prompt 'a cute cat'")
            sys.exit(1)
    elif args.prompt:
        # Single generation mode
        try:
            glb_path, stl_path = generate_3d_model(args.prompt, args.output)
            print(f"\n✨ Done! Check the {args.output} folder for your models.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    else:
        # Explicit interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()