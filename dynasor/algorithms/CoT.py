from .Reasoning import ReasoningProgram
from dynasor.clients import Client

class CoT(ReasoningProgram):
    def __init__(self, client: Client):
        super().__init__(client)

    def run(self, input_message):
        return self.client.chat_completions(messages=input_message)

        self.input_message = input_message
        self.results = []
        self.result = None
        for i in range(n_steps):
            self.results.append(self.client.chat.completions.create(messages=[{"role": "user", "content": self.input_message}]))
        self.result = self.results[-1]
        return self.result
