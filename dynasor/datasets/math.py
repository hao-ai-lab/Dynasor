from dynasor.datasets.dataloader import DatasetLoader
from dynasor.datasets.utils import load_jsonl
from datasets import load_dataset, Dataset
import os

# Define supported splits for each dataset
DATASET_SPLITS = {
    "gsm8k": ["train", "test"],
    "math": ["train", "test"], 
    "asdiv": ["train"],
    "qwen-AIME24": ["test"],
    "qwen-AMC23": ["test"]
}

DATASET_CONFIGS = {
    'gsm8k': {'loader': load_dataset, 'args': ('gsm8k', 'main'), 'split': 'split'},
    'math': {'loader': load_dataset, 'args': ('math', 'main')},
    'asdiv': {'loader': load_dataset, 'args': ('EleutherAI/asdiv')},
    'qwen-AIME24': {'loader': lambda x, split=None: Dataset.from_list(load_jsonl(x)), 'args': ('https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data/aime24/test.jsonl',)},
    'qwen-AMC23': {'loader': load_jsonl, 'args': ('https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data/amc23/test.jsonl',)}
}

class MathDatasetLoader(DatasetLoader):
    def __init__(self, dataset_path: str, split: str = "train"):
        if dataset_path in DATASET_SPLITS:
            self.dataset_name = dataset_path
            # Validate split is supported for this dataset
            if split not in DATASET_SPLITS[dataset_path]:
                raise ValueError(f"Dataset {dataset_path} only supports splits: {DATASET_SPLITS[dataset_path]}")
            self.split = split
        else:
            if os.path.isabs(dataset_path):
                dataset_name = os.path.basename(dataset_path).split(".")[0].lower()
            else:
                dataset_name = dataset_path.split("/")[-1].split(".")[0].lower()
        super().__init__(dataset_path)

    def load(self, num_samples: int = None):
        config = DATASET_CONFIGS.get(self.dataset_name)
        if not config:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        # Handle split parameter for dataset loaders that need it
        if 'split' in config:
            dataset = config['loader'](*config['args'], split=self.split)
        else:
            dataset = config['loader'](*config['args'])
        
        if num_samples is not None:
            if isinstance(dataset, list):
                dataset = dataset[:num_samples]
            else:
                # Assuming it's a Dataset object from datasets library
                dataset = dataset.select(range(num_samples))
        
        if isinstance(dataset, list):
            self._dataset = dataset
        else:
            # Convert Dataset to list of dictionaries
            self._dataset = [item for item in dataset]
    def process_dataset(self, dataset: list[dict]) -> list[dict]:
        return dataset

    def ground_truths(self) -> list[dict]:
        for item in self._dataset:
            item['answer'] = extract_answer(item['answer'], self.dataset_name)
        return self._dataset
