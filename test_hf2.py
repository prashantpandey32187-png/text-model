from gradio_client import Client
import sys
import json

print("Connecting to TRELLIS.2...", flush=True)
c = Client('prithivMLmods/TRELLIS.2-Text-to-3D')
print("Connected!", flush=True)

# Step 1: Text -> Image
print("\n=== Step 1: Text to Image ===", flush=True)
result1 = c.predict(
    "A cute cat 3D model",  # prompt (textbox id=7)
    api_name="/generate_txt2img"
)
print(f"Image result: {result1}", flush=True)
print(f"Type: {type(result1)}", flush=True)

# Save image
if isinstance(result1, dict) and 'path' in result1:
    import shutil
    shutil.copy(result1['path'], 'trellis_image.png')
    print(f"Copied image to trellis_image.png", flush=True)
elif isinstance(result1, str) and result1.startswith('http'):
    import requests
    r = requests.get(result1)
    with open('trellis_image.png', 'wb') as f:
        f.write(r.content)
    print(f"Downloaded image from {result1}", flush=True)
elif isinstance(result1, str):
    print(f"Is file path: {result1}", flush=True)
    import shutil, os
    if os.path.exists(result1):
        shutil.copy(result1, 'trellis_image.png')
        print(f"Copied to trellis_image.png", flush=True)