import requests, sys
import os

# Load token from environment variable
HF_TOKEN = os.getenv('HF_TOKEN', 'your_huggingface_token_here')
headers = {'Authorization': f'Bearer {HF_TOKEN}'}

print('=== Testing HF Token ===', flush=True)

# Test token
r = requests.get('https://huggingface.co/api/whoami', headers=headers)
print(f'whoami: {r.status_code}', flush=True)
if r.status_code == 200:
    print(f'User: {r.json().get("name", "?")}', flush=True)

print(), print('=== Testing Shap-E ===', flush=True)

# Test Shap-E
r2 = requests.post(
    'https://api-inference.huggingface.co/models/openai/shap-e',
    headers=headers,
    json={'inputs': 'A cute cat'}
)
print(f'Shap-E Status: {r2.status_code}', flush=True)
print(f'Shap-E Content-Type: {r2.headers.get("content-type", "?")}', flush=True)
print(f'Shap-E Response length: {len(r2.content)} bytes', flush=True)

if r2.status_code == 200:
    ct = r2.headers.get('content-type', '')
    if 'json' in ct:
        data = r2.json()
        print(f'JSON response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}', flush=True)
    else:
        # Save binary output
        with open('shap_e_output.glb', 'wb') as f:
            f.write(r2.content)
        print(f'Saved GLB file: shap_e_output.glb ({len(r2.content)} bytes)', flush=True)
else:
    print(f'Error: {r2.text[:500]}', flush=True)