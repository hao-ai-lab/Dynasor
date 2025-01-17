from openai import OpenAI
from abc import ABC, abstractmethod
import os

class Client(ABC):
    @abstractmethod
    def get_responses(self, messages, n_samples=3, **kwargs) -> list[str]:
        raise NotImplementedError

class OpenAIClient(Client):
    def __init__(
            self, 
            model,
            api_key,
            api_base,
        ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def chat_completions(self, messages, **kwargs) -> list[str]:
        responses = []
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        
        for choice in response.choices:
            responses.append(choice.message.content)
        return responses

class DeepSeekClient(OpenAIClient):
    def __init__(
            self, 
            model="deepseek-chat",
            api_key=None,
            api_base='https://api.deepseek.com',
        ):
        super().__init__(model, api_key, api_base)

    def get_responses(self, messages, **kwargs) -> list[str]:
        return self.chat_completions(messages, **kwargs)

class vllmClient(OpenAIClient):
    def __init__(
            self, 
            model,
            api_key=None,
            api_base='http://localhost:8000/v1',
        ):
        super().__init__(model, api_key, api_base)

class SGLangClient(OpenAIClient):
    def __init__(
            self, 
            model,
            api_key=None,
            api_base='http://localhost:30000',
        ):
        super().__init__(model, api_key, api_base)