from functools import partial
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, validator

from meerkat.engines.text.completion import TextCompletion
from meerkat.ops.watch.abstract import WatchLogger
from meerkat.tools.lazy_loader import LazyLoader

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
    def with_openai(cls, key: str):
        return OpenAIChatCompletion(key)


class MockChatCompletion(ChatCompletion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engine = lambda *args, **kwargs: "\n---Mock Response---"

    def response(self, response: str):
        self.engine = lambda *args, **kwargs: response
        return self

    def run(self, prompt: str) -> str:
        return prompt + self.engine()


class Message(BaseModel):
    role: str
    content: str

    @validator("role")
    def role_must_be_system_or_user(cls, v):
        if v not in ["system", "user"]:
            raise ValueError("role must be system or user")
        return v


class OpenAIChatCompletion(ChatCompletion):

    COST_PER_TOKEN = {
        "gpt-3.5-turbo": 0.002 / 1000,
    }

    def __init__(
        self,
        key: str = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ):
        self._check_import()
        self.engine = partial(openai.ChatCompletion.create, api_key=key)
        self.configure(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            n=n,
            stop=stop,
        )

    def _check_import(self):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI is not installed. Install with `pip install meerkat[openai]`."
            )

    def model(self, model: str):
        """The name of the model to use for completion."""
        self.engine = partial(self.engine, model=model)
        return self

    def temperature(self, temperature: float):
        """The temperature of the model. Higher values will result in more creative completions, but also more mistakes."""
        self.engine = partial(self.engine, temperature=temperature)
        return self

    def help(self):
        print(self.engine.__doc__)

    def set_logger(self, logger: WatchLogger):
        self.logger = logger

    def log_engine_run(
        self,
        errand_run_id: str,
        input: str,
        output: str,
        engine: str,
        cost: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Log the engine run."""
        self.logger.log_engine_run(
            errand_run_id,
            input=input,
            output=output,
            engine=engine,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def run(
        self,
        prompt: str,
        history: List[Message] = [],
        system_prompt: str = "You are a helpful assistant.",
    ):
        """Run the engine on a prompt."""
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
        self.response = response

        self.log_engine_run(
            input={
                "prompt": prompt,
                "history": history,
                "system_prompt": system_prompt,
            },
            output=response,
            engine=self.name,
            cost=self.COST_PER_TOKEN[self.name],
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
        )

        return response["choices"][0]["message"]["content"]


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
