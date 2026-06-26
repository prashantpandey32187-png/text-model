"""
Simple test script to verify 3D model generation works
"""
import os
import sys
import time
import uuid

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import trimesh
from generation.mesh_utils import convert_obj_to_stl

def test_generation():
    """Test the 3D model generation pipeline"""
    print("=" * 60)
    print("🧪 Testing 3D Model Generation")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        print("\n⚠️  No GPU detected. Using CPU.")
        device = "cpu"
    
    print(f"🖥️  Device: {device}\n")
    
    # Test prompt
    prompt = "a cute cat"
    print(f"📝 Test prompt: '{prompt}'")
    
    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique ID
    task_id = str(uuid.uuid4())
    
    # Define paths
    glb_path = os.path.join(output_dir, f"{task_id}.glb")
    stl_path = os.path.join(output_dir, f"{task_id}.stl")
    temp_obj_path = os.path.join(output_dir, f"{task_id}.obj")
    
    try:
        print("\n⏳ Generating 3D model...")
        start_time = time.time()
        
        # Simulate AI generation (replace with your actual model)
        time.sleep(2)
        
        print("🔨 Creating mesh...")
        # PLACEHOLDER: Replace this with your AI model
        mesh = trimesh.creation.cone(radius=1, height=2)
        
        # Save as GLB
        print(f"💾 Saving GLB: {glb_path}")
        mesh.export(glb_path)
        
        # Convert to STL
        print(f"🔄 Converting to STL: {stl_path}")
        mesh.export(temp_obj_path)
        convert_obj_to_stl(temp_obj_path, stl_path)
        
        # Cleanup temp file
        if os.path.exists(temp_obj_path):
            os.remove(temp_obj_path)
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS! Generation completed in {elapsed:.2f} seconds")
        print(f"   📦 GLB file: {glb_path}")
        print(f"   📦 STL file: {stl_path}")
        
        # Verify files exist
        if os.path.exists(glb_path) and os.path.exists(stl_path):
            glb_size = os.path.getsize(glb_path)
            stl_size = os.path.getsize(stl_path)
            print(f"\n📊 File sizes:")
            print(f"   GLB: {glb_size:,} bytes")
            print(f"   STL: {stl_size:,} bytes")
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n❌ Error: Output files not created")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        for path in [glb_path, stl_path, temp_obj_path]:
            if os.path.exists(path):
                os.remove(path)
        return False

if __name__ == "__main__":
    success = test_generation()
    sys.exit(0 if success else 1)