from dynasor.datasets.dataloader import DatasetLoader

class SimpleDatasetLoader(DatasetLoader):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

