import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Pollo AI API Configuration
POLLO_API_KEY = os.getenv("POLLO_API_KEY")
POLLO_BASE_URL = "https://pollo.ai/api/platform/generation/google/veo3-1"

# Validate API key is present
if not POLLO_API_KEY:
    raise ValueError(
        "POLLO_API_KEY not found in environment variables. "
        "Please set it in your .env file."
    )