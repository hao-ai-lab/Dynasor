from dynasor.datasets.dataloader import DatasetLoader
import os


class MathDatasetLoader(DatasetLoader):
    def __init__(self, dataset_path: str, split: str = "train"):
        supported_datasets = ["gsm8k", "MATH", "AIME24"]
        if dataset_path in supported_datasets:
            dataset_name = dataset_path
        else:
            if os.path.isabs(dataset_path):
                dataset_name = os.path.basename(dataset_path).split(".")[0].lower()
            else:
                dataset_name = dataset_path.split("/")[-1].split(".")[0].lower()
        super().__init__(dataset_path)

