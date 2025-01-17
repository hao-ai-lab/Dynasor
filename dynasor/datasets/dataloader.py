from dynasor.datasets.utils import load_jsonl
from typing import Callable
class DatasetLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    
    def load(self, n_rows: int = None, prepare_prompt: Callable = None):
        dataset = load_jsonl(self.dataset_path)
        if n_rows is not None:
            dataset = dataset[:n_rows]
        self._dataset = dataset
    
    def prepare_dataset(self, process_func: Callable):
        dataset = [process_func(item) for item in self._dataset]
        return dataset

    def __getitem__(self, index: int):
        return self.messages[index]
    
    def __len__(self):
        return len(self.messages)
    
    def __iter__(self):
        for item in self.messages:
            yield item

class AutoDatasetLoader(DatasetLoader):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    def prepare_prompt(self, item: dict) -> str:
        raise NotImplementedError
