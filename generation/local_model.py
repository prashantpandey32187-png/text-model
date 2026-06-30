"""
Local Text-to-3D Model Generator
Runs entirely locally without external API calls
"""
import os
import sys
import time
import torch
import trimesh
import numpy as np
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class LocalTextTo3D:
    """Generate 3D models locally using downloaded models"""
    
    def __init__(self):
        # Check for GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            # Use GPU 0 by default
            self.device = f"cuda:0"
            print(f"   🚀 GPU detected: {gpu_name}")
            print(f"   📊 Available GPUs: {gpu_count}")
            # Enable cuDNN benchmark for faster inference
            torch.backends.cudnn.benchmark = True
        else:
            self.device = "cpu"
            print(f"   ⚠️  No GPU detected. Running on CPU (slower).")
            print(f"   💡 Install CUDA-enabled PyTorch for faster generation.")
        
        self.model_cache_dir = Path.home() / ".cache" / "text-to-3d-models"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   🖥️  Device: {self.device}")
        print(f"   📁 Model cache: {self.model_cache_dir}")
    
    def generate(self, prompt: str, output_path: str) -> str:
        """
        Generate 3D model from text prompt using AI-powered generation
        
        Args:
            prompt: Text description
            output_path: Where to save the model
            
        Returns:
            Path to generated model
        """
        print(f"   🔨 Analyzing prompt: '{prompt}'")
        
        # Try AI-powered generation first (high quality, textured)
        try:
            print(f"   🎨 Using AI-powered generation (high quality)...")
            return self._generate_with_sd(prompt, output_path)
        except Exception as e:
            print(f"   ⚠️  AI generation failed: {e}")
            print(f"   🔄 Falling back to procedural generation...")
            return self._generate_procedural(prompt, output_path)
    
    def _generate_with_sd(self, prompt: str, output_path: str) -> str:
        """Generate using Stable Diffusion + depth-to-3D"""
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            from transformers import pipeline as transformers_pipeline
            import cv2
            
            print(f"   📦 Loading models (first run will download)...")
            
            # Load depth estimation model
            depth_estimator = transformers_pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if self.device.startswith("cuda") else -1
            )
            
            # Load Stable Diffusion
            sd_model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(
                sd_model_id,
                cache_dir=str(self.model_cache_dir),
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(self.device)
            
            # Generate image from prompt
            print(f"   🎨 Generating image from prompt...")
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            
            # Save intermediate image
            temp_image = output_path.replace('.glb', '_temp.png')
            image.save(temp_image)
            
            # Estimate depth
            print(f"   📐 Estimating depth...")
            depth_result = depth_estimator(temp_image)
            depth_image = depth_result['depth']
            depth_array = np.array(depth_image)
            
            # Convert depth to 3D mesh
            print(f"   🔄 Converting to 3D mesh...")
            mesh = self._depth_to_mesh(np.array(image), depth_array)
            
            # Save mesh
            mesh.export(output_path)
            
            # Cleanup
            if os.path.exists(temp_image):
                os.remove(temp_image)
            
            print(f"   ✅ 3D model generated from AI!")
            return output_path
            
        except ImportError as e:
            raise RuntimeError(f"Required libraries not installed: {e}")
        except Exception as e:
            raise RuntimeError(f"Local generation failed: {e}")
    
    def _depth_to_mesh(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> trimesh.Trimesh:
        """Convert RGB-D image to high-quality textured 3D mesh"""
        h, w = depth_map.shape
        
        print(f"   🔄 Processing {w}x{h} depth map to 3D mesh...")
        
        # Create vertex grid with higher resolution
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize coordinates
        x = (x - w/2) / (w/2)
        y = -(y - h/2) / (h/2)  # Flip Y
        z = -depth_map / 255.0 * 5  # Scale depth
        
        # Create vertices
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # Create faces (triangles)
        faces = []
        for i in range(h-1):
            for j in range(w-1):
                idx = i * w + j
                faces.append([idx, idx+1, idx+w])
                faces.append([idx+1, idx+w+1, idx+w])
        
        faces = np.array(faces)
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Apply texture from RGB image
        print(f"   🎨 Applying textures...")
        
        # Resize RGB image to match mesh resolution if needed
        if rgb_image.shape[:2] != (h, w):
            from PIL import Image
            rgb_pil = Image.fromarray(rgb_image)
            rgb_pil = rgb_pil.resize((w, h), Image.Resampling.LANCZOS)
            rgb_image = np.array(rgb_pil)
        
        # Create UV coordinates for texturing
        uv_coords = np.zeros((len(vertices), 2))
        uv_coords[:, 0] = (x.flatten() + 1) / 2  # U coordinate
        uv_coords[:, 1] = (y.flatten() + 1) / 2  # V coordinate
        
        # Set texture coordinates
        mesh.visual.uv = uv_coords
        
        # Create texture image
        from PIL import Image
        texture_image = Image.fromarray(rgb_image)
        
        # Set the texture
        mesh.visual.material.image = texture_image
        mesh.visual.material.diffuse = [255, 255, 255, 255]
        
        # Improve mesh quality
        print(f"   ✨ Enhancing mesh quality...")
        
        # Remove duplicate vertices
        mesh.merge_vertices()
        
        # Smooth the mesh slightly for better appearance
        mesh = mesh.smoothed()
        
        # Simplify mesh to reasonable polygon count (keep detail but reduce size)
        target_faces = min(10000, len(mesh.faces))
        if len(mesh.faces) > target_faces:
            mesh = mesh.simplify_quadric_decimation(target_faces)
        
        # Ensure mesh is watertight
        if not mesh.is_watertight:
            mesh.fill_holes()
        
        # Recalculate normals for proper lighting
        mesh.rezero()
        mesh.fix_normals()
        
        print(f"   ✅ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
    
    def _generate_procedural(self, prompt: str, output_path: str) -> str:
        """
        Generate procedural 3D model based on prompt keywords
        Creates different shapes based on what's in the prompt
        """
        prompt_lower = prompt.lower()
        
        print(f"   🎨 Creating procedural 3D model...")
        
        # Analyze prompt and create appropriate shape
        if any(word in prompt_lower for word in ['cat', 'dog', 'animal', 'pet', 'lion', 'tiger']):
            print(f"   🐱 Generating animal-like form...")
            mesh = self._create_animal_shape()
        elif any(word in prompt_lower for word in ['chair', 'seat', 'furniture']):
            print(f"   🪑 Generating furniture...")
            mesh = self._create_chair()
        elif any(word in prompt_lower for word in ['car', 'vehicle', 'auto', 'truck']):
            print(f"   🚗 Generating vehicle...")
            mesh = self._create_vehicle()
        elif any(word in prompt_lower for word in ['house', 'building', 'home', 'castle']):
            print(f"   🏠 Generating building...")
            mesh = self._create_building()
        elif any(word in prompt_lower for word in ['tree', 'plant', 'flower']):
            print(f"   🌳 Generating plant...")
            mesh = self._create_tree()
        elif any(word in prompt_lower for word in ['person', 'human', 'man', 'woman', 'character']):
            print(f"   🧑 Generating humanoid figure...")
            mesh = self._create_humanoid()
        elif any(word in prompt_lower for word in ['sphere', 'ball', 'globe', 'orb']):
            print(f"   ⚽ Generating sphere...")
            mesh = trimesh.creation.icosphere(subdivisions=3)
        elif any(word in prompt_lower for word in ['tower', 'skyscraper', 'tall']):
            print(f"   🏢 Generating tower...")
            mesh = self._create_tower()
        elif any(word in prompt_lower for word in ['sword', 'blade', 'weapon']):
            print(f"   ⚔️  Generating sword...")
            mesh = self._create_sword()
        elif any(word in prompt_lower for word in ['vase', 'pot', 'container']):
            print(f"   🏺 Generating vase...")
            mesh = self._create_vase()
        else:
            print(f"   📦 Generating generic 3D object...")
            mesh = self._create_generic_object()
        
        # Apply some random variation based on prompt
        seed = hash(prompt) % 1000
        np.random.seed(seed)
        
        # Add slight random rotation for variety
        rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        mesh.vertices = mesh.vertices @ rotation.T
        
        # Save
        mesh.export(output_path)
        print(f"   ✅ Procedural model created")
        return output_path
    
    def _create_animal_shape(self) -> trimesh.Trimesh:
        """Create an animal-like shape"""
        # Body
        body = trimesh.creation.capsule(radius=0.8, height=2.0)
        # Head
        head = trimesh.creation.icosphere(subdivisions=2, radius=0.6)
        head.apply_translation([0, 1.2, 0])
        # Legs
        leg1 = trimesh.creation.cylinder(radius=0.2, height=1.0)
        leg1.apply_translation([0.5, -0.8, -0.5])
        leg2 = trimesh.creation.cylinder(radius=0.2, height=1.0)
        leg2.apply_translation([0.5, -0.8, 0.5])
        leg3 = trimesh.creation.cylinder(radius=0.2, height=1.0)
        leg3.apply_translation([-0.5, -0.8, -0.5])
        leg4 = trimesh.creation.cylinder(radius=0.2, height=1.0)
        leg4.apply_translation([-0.5, -0.8, 0.5])
        
        return trimesh.util.concatenate([body, head, leg1, leg2, leg3, leg4])
    
    def _create_chair(self) -> trimesh.Trimesh:
        """Create a chair shape"""
        seat = trimesh.creation.box(extents=[1.5, 1.5, 0.2])
        seat.apply_translation([0, 0, 1.0])
        back = trimesh.creation.box(extents=[1.5, 0.2, 1.5])
        back.apply_translation([0, -0.65, 1.6])
        leg1 = trimesh.creation.cylinder(radius=0.1, height=1.0)
        leg1.apply_translation([0.6, 0.6, 0.5])
        leg2 = trimesh.creation.cylinder(radius=0.1, height=1.0)
        leg2.apply_translation([-0.6, 0.6, 0.5])
        leg3 = trimesh.creation.cylinder(radius=0.1, height=1.0)
        leg3.apply_translation([0.6, -0.6, 0.5])
        leg4 = trimesh.creation.cylinder(radius=0.1, height=1.0)
        leg4.apply_translation([-0.6, -0.6, 0.5])
        
        return trimesh.util.concatenate([seat, back, leg1, leg2, leg3, leg4])
    
    def _create_vehicle(self) -> trimesh.Trimesh:
        """Create a vehicle shape"""
        body = trimesh.creation.box(extents=[3.0, 1.5, 1.0])
        body.apply_translation([0, 0, 0.8])
        cabin = trimesh.creation.box(extents=[1.5, 1.3, 0.8])
        cabin.apply_translation([0.3, 0, 1.6])
        wheel1 = trimesh.creation.cylinder(radius=0.4, height=0.2, sections=16)
        wheel1.apply_translation([1.0, 0.8, 0.4])
        wheel2 = trimesh.creation.cylinder(radius=0.4, height=0.2, sections=16)
        wheel2.apply_translation([-1.0, 0.8, 0.4])
        wheel3 = trimesh.creation.cylinder(radius=0.4, height=0.2, sections=16)
        wheel3.apply_translation([1.0, -0.8, 0.4])
        wheel4 = trimesh.creation.cylinder(radius=0.4, height=0.2, sections=16)
        wheel4.apply_translation([-1.0, -0.8, 0.4])
        
        return trimesh.util.concatenate([body, cabin, wheel1, wheel2, wheel3, wheel4])
    
    def _create_building(self) -> trimesh.Trimesh:
        """Create a building shape"""
        main = trimesh.creation.box(extents=[2.0, 2.0, 3.0])
        main.apply_translation([0, 0, 1.5])
        roof = trimesh.creation.cone(radius=1.5, height=1.0, sections=4)
        roof.apply_translation([0, 0, 3.5])
        
        return trimesh.util.concatenate([main, roof])
    
    def _create_tree(self) -> trimesh.Trimesh:
        """Create a tree shape"""
        trunk = trimesh.creation.cylinder(radius=0.3, height=2.0, sections=8)
        trunk.apply_translation([0, 0, 1.0])
        foliage = trimesh.creation.icosphere(subdivisions=2, radius=1.2)
        foliage.apply_translation([0, 0, 2.8])
        
        return trimesh.util.concatenate([trunk, foliage])
    
    def _create_humanoid(self) -> trimesh.Trimesh:
        """Create a humanoid figure"""
        torso = trimesh.creation.capsule(radius=0.5, height=1.5)
        torso.apply_translation([0, 0, 1.8])
        head = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        head.apply_translation([0, 0, 3.0])
        arm1 = trimesh.creation.capsule(radius=0.15, height=1.2)
        arm1.apply_translation([0.7, 0, 2.2])
        arm2 = trimesh.creation.capsule(radius=0.15, height=1.2)
        arm2.apply_translation([-0.7, 0, 2.2])
        leg1 = trimesh.creation.capsule(radius=0.2, height=1.5)
        leg1.apply_translation([0.3, 0, 0.8])
        leg2 = trimesh.creation.capsule(radius=0.2, height=1.5)
        leg2.apply_translation([-0.3, 0, 0.8])
        
        return trimesh.util.concatenate([torso, head, arm1, arm2, leg1, leg2])
    
    def _create_tower(self) -> trimesh.Trimesh:
        """Create a tower shape"""
        base = trimesh.creation.cylinder(radius=1.2, height=0.5, sections=8)
        base.apply_translation([0, 0, 0.25])
        tower = trimesh.creation.cylinder(radius=0.8, height=4.0, sections=8)
        tower.apply_translation([0, 0, 2.5])
        top = trimesh.creation.cone(radius=1.0, height=1.0, sections=8)
        top.apply_translation([0, 0, 5.0])
        
        return trimesh.util.concatenate([base, tower, top])
    
    def _create_sword(self) -> trimesh.Trimesh:
        """Create a sword shape"""
        blade = trimesh.creation.box(extents=[0.2, 0.8, 3.0])
        blade.apply_translation([0, 0, 1.5])
        handle = trimesh.creation.cylinder(radius=0.15, height=1.0, sections=8)
        handle.apply_translation([0, 0, -0.3])
        guard = trimesh.creation.box(extents=[1.0, 0.2, 0.2])
        guard.apply_translation([0, 0, -0.7])
        
        return trimesh.util.concatenate([blade, handle, guard])
    
    def _create_vase(self) -> trimesh.Trimesh:
        """Create a vase shape"""
        # Create vase using lathe
        profile = [
            [0, 0],
            [0.3, 0],
            [0.3, 0.5],
            [0.5, 1.0],
            [0.4, 2.0],
            [0.6, 2.5],
            [0.5, 3.0],
            [0.3, 3.5],
            [0, 3.5]
        ]
        vase = trimesh.creation.lathe(profile, sections=16)
        
        return vase
    
    def _create_generic_object(self) -> trimesh.Trimesh:
        """Create a generic interesting shape"""
        # Combine multiple primitives
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        box = trimesh.creation.box(extents=[1.5, 1.5, 1.5])
        
        # Boolean intersection
        mesh = sphere.intersection(box)
        
        if mesh.is_empty:
            mesh = sphere
        
        return mesh


def generate_local_3d(prompt: str, output_path: str) -> str:
    """
    Convenience function for local 3D generation
    
    Args:
        prompt: Text description
        output_path: Where to save
        
    Returns:
        Path to generated model
    """
    generator = LocalTextTo3D()
    return generator.generate(prompt, output_path)


if __name__ == "__main__":
    # Test
    test_prompt = "a cute cat"
    test_output = "test_local.glb"
    
    print("=" * 60)
    print("Testing Local Text-to-3D Generator")
    print("=" * 60)
    
    generator = LocalTextTo3D()
    result = generator.generate(test_prompt, test_output)
    
    print(f"\n✅ Test complete! Output: {result}")