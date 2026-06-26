import sys
from gradio_client import Client

print("Starting...", flush=True)

c = Client('prithivMLmods/TRELLIS.2-Text-to-3D', verbose=True)
print("Connected to TRELLIS.2", flush=True)

# Try the simplest API call
print("Making prediction...", flush=True)
try:
    result = c.predict(
        "A cute cat",
        api_name="/predict"
    )
    print(f"Result: {result}", flush=True)
    print(f"Type: {type(result)}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    # Try alternative: Run the predict without api_name
    try:
        result = c.predict("A cute cat", "A cute cat")
        print(f"Result2: {result}", flush=True)
    except Exception as e2:
        print(f"Error2: {e2}", flush=True)

print("Done", flush=True)