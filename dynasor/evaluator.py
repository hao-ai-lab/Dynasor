from dynasor.utils import load_jsonl

class DataLoader:
    def __init__(self, path: str):
        self.data = load_jsonl(path)

    def __len__(self):
        return len(self.data)
