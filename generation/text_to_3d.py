"""
Text-to-3D Model Generation Module
Supports multiple backends: HuggingFace API, Pollo AI, or local models
"""
import os
import sys
import time
import requests
import trimesh
import numpy as np
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class TextTo3DGenerator:
    """Generate 3D models from text prompts"""
    
    def __init__(self, backend='huggingface'):
        """
        Initialize the generator with specified backend
        
        Args:
            backend: 'huggingface', 'pollo', or 'local'
        """
        self.backend = backend
        self.hf_token = os.getenv('HF_TOKEN')
        
        if backend == 'huggingface' and not self.hf_token:
            print("⚠️  Warning: HF_TOKEN not found in environment. Using placeholder mode.")
            self.backend = 'placeholder'
    
    def generate(self, prompt: str, output_path: str) -> str:
        """
        Generate a 3D model from text prompt
        
        Args:
            prompt: Text description of the 3D model
            output_path: Path to save the generated model
            
        Returns:
            Path to the generated model
        """
        print(f"   Backend: {self.backend}")
        
        if self.backend == 'huggingface':
            return self._generate_huggingface(prompt, output_path)
        elif self.backend == 'pollo':
            return self._generate_pollo(prompt, output_path)
        else:
            return self._generate_placeholder(prompt, output_path)
    
    def _generate_huggingface(self, prompt: str, output_path: str) -> str:
        """Generate using HuggingFace Shap-E or TRELLIS model"""
        print(f"   🚀 Using HuggingFace API...")
        
        # Try Shap-E model first
        api_url = "https://api-inference.huggingface.co/models/openai/shap-e"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        print(f"   📡 Calling Shap-E model...")
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": prompt},
            timeout=120
        )
        
        if response.status_code == 200:
            # Save the GLB file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✅ Model generated successfully!")
            return output_path
        else:
            # If Shap-E fails, try TRELLIS
            print(f"   ⚠️  Shap-E returned {response.status_code}, trying TRELLIS...")
            return self._generate_trellis(prompt, output_path)
    
    def _generate_trellis(self, prompt: str, output_path: str) -> str:
        """Generate using TRELLIS model via Gradio"""
        try:
            from gradio_client import Client
            
            print(f"   📡 Connecting to TRELLIS.2...")
            client = Client('prithivMLmods/TRELLIS.2-Text-to-3D', verbose=False)
            
            print(f"   🔨 Generating 3D model...")
            result = client.predict(
                prompt,
                api_name="/generate_txt2img"
            )
            
            # Handle different result formats
            if isinstance(result, dict) and 'path' in result:
                import shutil
                shutil.copy(result['path'], output_path)
            elif isinstance(result, str):
                if result.startswith('http'):
                    # Download from URL
                    response = requests.get(result)
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                else:
                    # Copy local file
                    import shutil
                    shutil.copy(result, output_path)
            
            print(f"   ✅ Model generated successfully!")
            return output_path
            
        except Exception as e:
            print(f"   ❌ TRELLIS generation failed: {e}")
            raise RuntimeError(f"Failed to generate 3D model: {e}")
    
    def _generate_pollo(self, prompt: str, output_path: str) -> str:
        """Generate using Pollo AI API"""
        try:
            from .api_client import PolloAIClient
            
            print(f"   🚀 Using Pollo AI...")
            client = PolloAIClient()
            
            # Generate video (Pollo is video-focused, but we can extract frames)
            video_bytes = client.text_to_video(
                prompt=prompt,
                duration=4,
                resolution='720p'
            )
            
            # Save video temporarily
            temp_video = output_path.replace('.glb', '.mp4')
            with open(temp_video, 'wb') as f:
                f.write(video_bytes)
            
            # For now, just save as-is (you'd need video-to-3D conversion)
            print(f"   ⚠️  Pollo generates video, not 3D models directly")
            print(f"   💾 Saved video to: {temp_video}")
            return temp_video
            
        except Exception as e:
            print(f"   ❌ Pollo generation failed: {e}")
            raise
    
    def _generate_placeholder(self, prompt: str, output_path: str) -> str:
        """
        Generate a placeholder mesh based on prompt keywords
        This is a fallback when no API is available
        """
        print(f"   ⚠️  Using placeholder mode (no API configured)")
        print(f"   💡 To generate real models, set HF_TOKEN in .env file")
        
        # Create different shapes based on prompt keywords
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['cat', 'dog', 'animal', 'pet']):
            print(f"   🐱 Creating animal-like shape...")
            mesh = trimesh.creation.cone(radius=1, height=2, sections=8)
        elif any(word in prompt_lower for word in ['chair', 'seat', 'furniture']):
            print(f"   🪑 Creating chair-like shape...")
            mesh = trimesh.creation.box(extents=[2, 2, 1])
        elif any(word in prompt_lower for word in ['car', 'vehicle', 'auto']):
            print(f"   🚗 Creating vehicle-like shape...")
            mesh = trimesh.creation.box(extents=[3, 1.5, 1])
        elif any(word in prompt_lower for word in ['sphere', 'ball', 'globe']):
            print(f"   ⚽ Creating sphere...")
            mesh = trimesh.creation.icosphere(subdivisions=2)
        elif any(word in prompt_lower for word in ['tower', 'building', 'skyscraper']):
            print(f"   🏢 Creating tower-like shape...")
            mesh = trimesh.creation.cylinder(radius=1, height=4, sections=8)
        else:
            print(f"   📦 Creating generic shape...")
            mesh = trimesh.creation.cone(radius=1, height=2)
        
        # Save the mesh
        mesh.export(output_path)
        print(f"   ✅ Placeholder model created")
        return output_path


def generate_3d_from_text(prompt: str, output_path: str, backend: str = 'huggingface') -> str:
    """
    Convenience function to generate 3D model from text
    
    Args:
        prompt: Text description
        output_path: Where to save the model
        backend: Which backend to use ('huggingface', 'pollo', 'local')
        
    Returns:
        Path to generated model
    """
    generator = TextTo3DGenerator(backend=backend)
    return generator.generate(prompt, output_path)


if __name__ == "__main__":
    # Test the generator
    test_prompt = "a cute cat"
    test_output = "test_output.glb"
    
    print("=" * 60)
    print("Testing Text-to-3D Generator")
    print("=" * 60)
    
    generator = TextTo3DGenerator()
    result = generator.generate(test_prompt, test_output)
    
    print(f"\n✅ Test complete! Output: {result}")