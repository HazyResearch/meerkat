import json
from typing import List, Optional, Union

from pydantic import BaseModel, validator

from meerkat.engines.text.completion import TextCompletion
from meerkat.ops.watch.abstract import WatchLogger
from meerkat.tools.lazy_loader import LazyLoader

from ..providers import OpenAIMixin

manifest = LazyLoader("manifest")
openai = LazyLoader("openai")
anthropic = LazyLoader("anthropic")


class ChatCompletion(TextCompletion):
    """
    A chat completion engine that takes in a message
    history and returns a completion.
    """

    @classmethod
    def with_langchain(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def with_mock(cls, **kwargs):
        return MockChatCompletion(**kwargs)

    @classmethod
    def with_openai(cls, key: str = None, organization: str = None):
        return OpenAIChatCompletion(key=key, organization=organization)

    def set_logger(self, logger: WatchLogger):
        self._logger = logger

    def set_errand_run_id(self, errand_run_id: str):
        self._errand_run_id = errand_run_id


class MockChatCompletion(ChatCompletion):
    """
    A text completion engine that takes in a prompt and returns a completion.
    """

    @property
    def parameter_mapping(self):
        """Map from Meerkat parameter names to the engine's parameter names."""
        return {**super().parameter_mapping, "dummy": "dummy"}

    def setup_engine(self, **kwargs):
        def mock(prompt, dummy="Mock response.") -> str:
            return dummy

        self._engine = mock

    def _check_import(self):
        pass

    def dummy(self, dummy: str):
        return self.configure(dummy=dummy)

    def parse_response(self, response: str) -> str:
        return response


class Message(BaseModel):
    role: str
    content: str

    @validator("role")
    def role_must_be_system_or_user(cls, v):
        if v not in ["system", "user"]:
            raise ValueError("role must be system or user")
        return v


class OpenAIChatCompletion(OpenAIMixin, ChatCompletion):

    COST_PER_TOKEN = {
        "gpt-3.5-turbo": 0.002 / 1000,
    }

    def __init__(
        self,
        key: Optional[str] = None,
        organization: Optional[str] = None,
        model: Optional[str] = "gpt-3.5-turbo",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 20,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ):
        self._organization = organization

        super().__init__(
            key=key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            n=n,
            stop=stop,
        )

    def setup_engine(self, key: str = None):
        self.authenticate(key)
        self._engine = openai.ChatCompletion.create

    def _check_import(self):
        try:
            import openai
        except ImportError:
            raise ImportError(
                f"{self.__class__.__name__} requires the openai package."
                " Please install it with `pip install openai`."
            )

    def run(
        self,
        prompt: str,
        history: List[Message] = [],
        system_prompt: str = "You are a helpful assistant.",
        skip_cache: bool = False
    ):
        """Run the engine on a prompt."""
        if not skip_cache:
            cached_run = self.on_run_start(prompt=prompt)
            if cached_run:
                return cached_run.output

        messages = (
            [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ]
            + [message.dict() for message in history]
            + [{"role": "user", "content": prompt}]
        )
        response = self.engine(messages=messages)
        result = self.parse_response(response)
        self.on_run_end(
            prompt=prompt,
            result=result,
            response=response,
        )
        return result

    def parse_response(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def on_run_end(self, prompt: str, result: str, response: dict):
        """Run after the engine has been run."""
        if self._logger is None:
            return prompt

        self._logger.log_engine_run(
            errand_run_id=self._errand_run_id
            if hasattr(self, "_errand_run_id")
            else None,
            input=prompt,
            output=result,
            engine=f"{self.name}/{self._model}",
            cost=self.COST_PER_TOKEN[self._model] * response["usage"]["total_tokens"],
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
            configuration=self.configuration
        )

    def key(self, key: str):
        self.setup_engine(key=key)
        return self


class PersonalityChatbot(ChatCompletion):
    def __init__(
        self,
        name: str,
        personality: str,
        bio: str,
    ):
        self.name = "Chatty the Chatbot"
        self.personality = "You are a helpful assistant."
        self.bio = "I am a chatbot."

    def personality(self, personality: str):
        self.personality = personality
        return self

    def description(self, description: str):
        self.description = description
        return self

    def run(
        self,
        prompt: str,
        history: List[Message] = [],
    ):
        """Run the engine on a prompt."""
        messages = (
            [
                {
                    "role": "system",
                    "content": self.description,
                }
            ]
            + [message.dict() for message in history]
            + [{"role": "user", "content": prompt}]
        )
        response = personality_chatbot(messages=messages)
        self.response = response
        return response["choices"][0]["message"]["content"]
