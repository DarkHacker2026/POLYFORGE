import json
import urllib.request
import os
from pathlib import Path
import sys

ROOT = Path(r"c:\Users\Dark Hacker\Desktop\hackathon project")
sys.path.insert(0, str(ROOT))
from grow_compiler import load_dotenv

def check_model():
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FIREWORKS_API_KEY")
    
    if not api_key:
        print("Error: FIREWORKS_API_KEY not found.")
        return

    req_model = "accounts/fireworks/models/kimi2.6"
    payload = {
        "model": req_model,
        "messages": [{"role": "user", "content": "Just say hello"}],
    }
    
    print(f"Requesting model: {req_model}")
    
    req = urllib.request.Request(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            print(f"Fireworks returned model: {body.get('model')}")
            # print("Full raw response keys:", list(body.keys()))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode()}")

if __name__ == "__main__":
    check_model()
