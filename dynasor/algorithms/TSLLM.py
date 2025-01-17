from dynasor.clients import Client
from dynasor.algorithms.Reasoning import ReasoningProgram

class RAP(ReasoningProgram):
    def __init__(self, client: Client):
        super().__init__(client)
        self.knobs["n_steps"] = 10
    
    def run(self, input_message, n_steps=10):
        self.input_message = input_message
        self.results = []
        self.result = None
        for i in range(n_steps):
            self.results.append(self.client.chat.completions.create(messages=[{"role": "user", "content": self.input_message}]))
        self.result = self.results[-1]
        return self.result