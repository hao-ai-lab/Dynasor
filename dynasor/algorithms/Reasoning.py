from abc import ABC, abstractmethod
from dynasor.clients import Client
from dynasor.datasets import DatasetLoader
from concurrent.futures import ThreadPoolExecutor

class ReasoningProgram:
    def __init__(self, client: Client):
        self.client = client
        self.knobs = {}
        self.state = dict(
            terminated=False,
        )
        self.certitude = {}
        
        self.input_message = None
        self.results = []
        self.result = None
        pass

    @property
    def is_terminated(self):
        return self.state['terminated']

    @abstractmethod
    def init_resource(self):
        raise NotImplementedError

    @abstractmethod
    def update_certitude(self):
        raise NotImplementedError
    
    @abstractmethod
    def expand(self):
        raise NotImplementedError
    
    @abstractmethod
    def aggregate(self, results: list[str]):
        raise NotImplementedError
    
    def run(self):
        # TODO(GindaChen&Yichao) This is a default implementation. Different reasoning programs may have different run function, but let's assume it is simple and consistent in base class.
        self.init_resource()
        
        while not self.is_terminated:
            self.expand()
            self.update_certitude()
        
        return self.aggregate()
    
    @classmethod
    def run_batch(
        cls, app: Client, dataset: DatasetLoader, is_parallel=False
    ):
        programs = [cls(dataset[i]) for i in range(len(dataset))]

        if is_parallel:
            with ThreadPoolExecutor() as executor:
                futures = []
                for program in programs:
                    futures.append(executor.submit(program.run, app))
                results = [future.result() for future in futures]
        else:
            results = []
            for program in programs:
                results.append(program.run(app))

        return results
