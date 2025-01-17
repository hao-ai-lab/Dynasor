import json

def load_jsonl(path: str) -> list[dict]:
    try:
        with open(path, "r") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        if path.startswith('http'):
            import requests
            import os
            local_path = os.path.basename(path)
            try:
                response = requests.get(path)
                response.raise_for_status()
                with open(local_path, 'w') as f:
                    f.write(response.text)
                with open(local_path, 'r') as f:
                    return [json.loads(line) for line in f]
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to download file from {path}: {str(e)}")
        else:
            raise
