from typing import Callable, Union, List
from meerkat.tools.lazy_loader import LazyLoader
from functools import partial

manifest = LazyLoader("manifest")
openai = LazyLoader("openai")
anthropic = LazyLoader("anthropic")


class BaseEngine:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Engine must implement the __call__ method.")


class EngineResponse:
    pass


class TextCompletion(BaseEngine):
    """
    A text completion engine that takes in a prompt
    and returns a completion.
    """

    @classmethod
    def with_anthropic(cls, key: str):
        return AnthropicTextCompletion(key)

    @classmethod
    def with_openai(cls, key: str):
        return OpenAITextCompletion(key)

    def model(self, model: str):
        """The name of the model to use for completion."""
        raise NotImplementedError("Engine must implement the model method.")

    def temperature(self, temperature: float):
        """The temperature of the model. Higher values will result in more creative completions, but also more mistakes."""
        raise NotImplementedError("Engine must implement the temperature method.")

    def max_tokens(self, max_tokens: int):
        """The maximum number of tokens to generate."""
        raise NotImplementedError("Engine must implement the max_tokens method.")

    def top_k(self, top_k: int):
        """The number of top tokens to consider when sampling."""
        raise NotImplementedError("Engine must implement the top_k method.")

    def top_p(self, top_p: float):
        """Nucleus sampling: the cumulative probability of candidates considered for sampling."""
        raise NotImplementedError("Engine must implement the top_p method.")

    def n(self, n: int):
        """The number of completions to generate."""
        raise NotImplementedError("Engine must implement the n method.")

    def stop(self, stop: Union[str, List[str]]):
        """The stop sequences."""
        raise NotImplementedError("Engine must implement the stop method.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class OpenAITextCompletion(TextCompletion):
    """
    A text completion engine that takes in a prompt
    and returns a completion.
    """

    def __init__(
            self,
            key: str,
        ):
        self._check_import()
        self.engine = partial(openai.Completion.create, api_key=key)

    def help(self):
        print(self.engine.__doc__)

    def _check_import(self):
        try:
            assert openai
        except ImportError:
            raise ImportError(
                f"{self.__class__.__name__} requires the openai package."
                " Please install it with `pip install openai`."
            )

    def model(self, model: str):
        """The name of the model to use for completion."""
        self.engine = partial(self.engine, model=model)
        return self

    def temperature(self, temperature: float):
        """The temperature of the model. Higher values will result in more creative completions, but also more mistakes."""
        self.engine = partial(self.engine, temperature=temperature)
        return self

    def max_tokens(self, max_tokens: int):
        """The maximum number of tokens to generate."""
        self.engine = partial(self.engine, max_tokens=max_tokens)
        return self

    def top_k(self, top_k: int):
        """The number of top tokens to consider when sampling."""
        self.engine = partial(self.engine, top_p=top_k)
        return self

    def top_p(self, top_p: float):
        """Nucleus sampling: the cumulative probability of candidates considered for sampling."""
        self.engine = partial(self.engine, top_p=top_p)
        return self

    def n(self, n: int):
        """The number of completions to generate."""
        self.engine = partial(self.engine, n=n)
        return self

    def stop(self, stop: Union[str, List[str]]):
        """The stop sequences."""
        self.engine = partial(self.engine, stop=stop)
        return self

    def run(self, prompt: str):
        """Run the engine on a prompt and return the completion."""
        response = self.engine(prompt=prompt)
        self.response = response
        return response.choices[0].text

    async def arun(self, prompt: str):
        return await self.run(prompt=prompt)


class AnthropicTextCompletion(TextCompletion):
    """
    A text completion engine that takes in a prompt
    and returns a completion.
    """

    def __init__(self, key: str):
        self.client = anthropic.Client(api_key=key)
        self.engine = self.client.completion

    @property
    def client(self):
        return self.client

    def model(self, model: str):
        """The name of the model to use for completion."""
        self.engine = partial(self.engine, model=model)
        return self

    def temperature(self, temperature: float):
        """The temperature of the model. Higher values will result in more creative completions, but also more mistakes."""
        self.engine = partial(self.engine, temperature=temperature)
        return self

    def max_tokens(self, max_tokens: int):
        """The maximum number of tokens to generate."""
        self.engine = partial(self.engine, max_tokens_to_sample=max_tokens)
        return self

    def top_k(self, top_k: int):
        """The number of top tokens to consider when sampling."""
        self.engine = partial(self.engine, top_k=top_k)
        return self

    def top_p(self, top_p: float):
        """Nucleus sampling: the cumulative probability of candidates considered for sampling."""
        self.engine = partial(self.engine, top_p=top_p)
        return self

    def n(self, n: int):
        """The number of completions to generate."""
        self.engine = partial(self.engine, n=n)
        return self

    def stop(self, stop: Union[str, List[str]]):
        """The stop sequences."""
        self.engine = partial(self.engine, stop_sequences=stop)
        return self

    def run(self, prompt: str):
        """Run the engine on a prompt and return the completion."""
        response = self.engine(prompt=prompt)
        self.response = response
        return response.choices[0].text

    async def arun(self, prompt: str):
        return await self.run(prompt=prompt)


"""
from meerkat.engines import TextCompletion

engine = TextCompletion.openai("davinci-003", temperature=0.1)
TextCompletion.openai().temperature(0.1).max_tokens(100).run("This is a test of the emergency broadcast system.")
completion = engine("This is a test of the emergency broadcast system.")
engine.response
"""
