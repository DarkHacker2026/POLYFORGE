import os
import urllib.request
import json
from pathlib import Path
from grow_compiler import load_dotenv

ROOT = Path(".")
load_dotenv(ROOT / ".env")
api_key = os.environ.get("FIREWORKS_API_KEY")

req = urllib.request.Request(
    "https://api.fireworks.ai/inference/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        models = data.get("data", [])
        print(f"Found {len(models)} models.")
        for m in models:
            print(m.get("id"))
except Exception as e:
    print("Error:", e)
