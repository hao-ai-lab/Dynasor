from dynasor.datasets.dataloader import DatasetLoader

class ArcAgiDatasetLoader(DatasetLoader):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

