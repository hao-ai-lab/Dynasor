from .Reasoning import ReasoningProgram
from dynasor.clients import Client

class ReAct(ReasoningProgram):
    def __init__(self, client: Client):
        super().__init__(client)

    def run(self, input_message, n_steps=10):
        pass