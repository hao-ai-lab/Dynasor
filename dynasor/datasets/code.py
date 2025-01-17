from dynasor.datasets.dataloader import DatasetLoader

class CodeDatasetLoader(DatasetLoader):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    def prepare_prompt(self, item: dict) -> str:
        raise NotImplementedError