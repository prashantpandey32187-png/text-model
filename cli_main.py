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

def generate_3d_model(prompt: str, output_dir: str = "outputs") -> tuple:
    """
    Generates a 3D model from text prompt using local GPU.
    Returns paths to both GLB and STL files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Determine device
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Using device: {device}")
    print(f"📝 Prompt: '{prompt}'")
    print(f"⏳ Generating 3D model...\n")
    
    # Define output paths
    glb_path = os.path.join(output_dir, f"{task_id}.glb")
    stl_path = os.path.join(output_dir, f"{task_id}.stl")
    
    try:
        # --- AI MODEL GENERATION ---
        # This is where your actual AI model would generate the mesh
        # Currently using a placeholder (cone) for demonstration
        time.sleep(3)  # Simulate GPU processing time
        
        print("🔨 Creating mesh...")
        # Replace this with your actual AI model inference
        # For now, creating a sample mesh based on prompt length (just for demo)
        mesh = trimesh.creation.cone(radius=1, height=2)
        
        # --- SAVE AS GLB ---
        print(f"💾 Saving GLB file: {glb_path}")
        mesh.export(glb_path)
        
        # --- CONVERT AND SAVE AS STL ---
        print(f"🔄 Converting to STL: {stl_path}")
        # Export as OBJ first (temporary), then convert to STL
        temp_obj_path = os.path.join(output_dir, f"{task_id}.obj")
        mesh.export(temp_obj_path)
        convert_obj_to_stl(temp_obj_path, stl_path)
        
        # Clean up temporary OBJ file
        if os.path.exists(temp_obj_path):
            os.remove(temp_obj_path)
        
        print(f"\n✅ Success! 3D model generated.")
        print(f"   📦 GLB: {glb_path}")
        print(f"   📦 STL: {stl_path}")
        
        return glb_path, stl_path
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
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
        default='outputs',
        help='Output directory for generated models (default: outputs)'
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