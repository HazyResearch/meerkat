from functools import partial
from typing import Callable, List, Optional, Union

from meerkat.engines.abstract import BaseEngine
from meerkat.ops.watch.abstract import WatchLogger
from meerkat.tools.lazy_loader import LazyLoader

manifest = LazyLoader("manifest")
openai = LazyLoader("openai")
anthropic = LazyLoader("anthropic")


class TextCompletion(BaseEngine):
    """
    A text completion engine that takes in a prompt and returns a completion.
    """

    _engine: Callable = None
    _logger: Optional[WatchLogger] = None

    def __init__(
        self,
        key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ):
        # Don't allow instantiation of the base class.
        if type(self) is TextCompletion:
            raise NotImplementedError(
                "The TextCompletion class is not meant to be instantiated directly."
                "Use the .with_* class methods e.g. TextCompletion.with_openai(...)."
            )
        self._check_import()
        self.setup_engine(key=key)

        self.configure(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            n=n,
            stop=stop,
        )

        self._key = key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_k = top_k
        self._top_p = top_p
        self._n = n
        self._stop = stop

    @property
    def configuration(self):
        return dict(
            key=self._key,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_k=self._top_k,
            top_p=self._top_p,
            n=self._n,
            stop=self._stop,
        )

    def _check_import(self):
        raise NotImplementedError(
            "The _check_import method must be implemented by a subclass."
        )

    def setup_engine(self, **kwargs):
        raise NotImplementedError(
            "The setup_engine method must be implemented by a subclass."
        )

    @property
    def engine(self):
        return self._engine

    def help(self):
        print(self.engine.__doc__)

    @classmethod
    def with_anthropic(cls, key: str = None):
        return AnthropicTextCompletion(key=key)

    @classmethod
    def with_mock(cls, **kwargs):
        return MockTextCompletion(**kwargs)

    @classmethod
    def with_openai(cls, key: str = None):
        return OpenAITextCompletion(key=key)

    @property
    def parameter_mapping(self):
        """Map from Meerkat parameter names to the engine's parameter names."""
        return {
            "model": "model",
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_k": "top_k",
            "top_p": "top_p",
            "n": "n",
            "stop": "stop",
        }

    def configure(self, **kwargs):
        kwargs = {
            self.parameter_mapping[k]: v for k, v in kwargs.items() if v is not None
        }
        self._engine = partial(self._engine, **kwargs)
        return self

    def model(self, model: str):
        """The name of the model to use for completion."""
        return self.configure(model=model)

    def temperature(self, temperature: float):
        """
        The temperature of the model. Higher values will result in
        more creative completions, but also more mistakes.
        """
        return self.configure(temperature=temperature)

    def max_tokens(self, max_tokens: int):
        """The maximum number of tokens to generate."""
        return self.configure(max_tokens=max_tokens)

    def top_k(self, top_k: int):
        """The number of top tokens to consider when sampling."""
        return self.configure(top_k=top_k)

    def top_p(self, top_p: float):
        """
        Nucleus sampling: the cumulative probability of candidates
        considered for sampling.
        """
        return self.configure(top_p=top_p)

    def n(self, n: int):
        """The number of completions to generate."""
        return self.configure(n=n)

    def stop(self, stop: Union[str, List[str]]):
        """The stop sequences."""
        return self.configure(stop=stop)

    def key(self, key: str):
        """The API key to use for the engine."""
        raise NotImplementedError("The key method must be implemented by a subclass.")

    def format_prompt(self, prompt: str):
        """Format the prompt for the engine."""
        return prompt

    def run(self, prompt: str) -> str:
        """Run the engine on a prompt and return the completion."""
        self.prompt = prompt
        self.response = self.engine(prompt=self.format_prompt(prompt))
        self.result = self.parse_response(self.response)
        self.on_run_end()
        return self.result

    def set_logger(self, logger: WatchLogger):
        self._logger = logger

    def set_errand_run_id(self, errand_run_id: str):
        self._errand_run_id = errand_run_id

    def on_run_end(self):
        """Run after the engine has been run."""
        if self._logger is None:
            return

        self._logger.log_engine_run(
            errand_run_id=self._errand_run_id
            if hasattr(self, "_errand_run_id")
            else None,
            input=self.prompt,
            output=self.result,
            engine=f"{self.name}/{self._model}",
            cost=0,
            input_tokens=0,
            output_tokens=0,
        )

    def parse_response(self, response) -> str:
        """Parse the response from the engine."""
        raise NotImplementedError(
            "The parse_response method must be implemented by a subclass."
        )

    async def arun(self, prompt: str) -> str:
        """Run the engine on a prompt asynchronously."""
        return self.run(prompt)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AnthropicTextCompletion(TextCompletion):
    """
    A text completion engine that takes in a prompt and returns a completion.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        model: Optional[str] = "claude-instant-v1",
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = 20,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ):
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

    def _check_import(self):
        try:
            assert anthropic
        except ImportError:
            raise ImportError(
                f"{self.__class__.__name__} requires the anthropic package."
                " Please install it with `pip install anthropic`."
            )

    @property
    def client(self):
        return self._client

    @property
    def parameter_mapping(self):
        """Map from Meerkat parameter names to the engine's parameter names."""
        return {
            "model": "model",
            "temperature": "temperature",
            "max_tokens": "max_tokens_to_sample",
            "top_k": "top_k",
            "top_p": "top_p",
            "n": "n",
            "stop": "stop",
        }

    def setup_engine(self, **kwargs):
        self._client = anthropic.Client(api_key=kwargs["key"])
        self._engine = self.client.completion

    def key(self, key: str):
        self.setup_engine(key=key)
        return self

    def format_prompt(self, prompt: str):
        return f"""\
{anthropic.HUMAN_PROMPT}
{prompt}
{anthropic.AI_PROMPT}\
"""

    def parse_response(self, response):
        try:
            return response["completion"]
        except KeyError:
            raise RuntimeError(f"Anthropic returned an invalid response: {response}")


class MockTextCompletion(TextCompletion):
    """
    A text completion engine that takes in a prompt and returns a completion.
    """

    def setup_engine(self, **kwargs):
        def mock(prompt, response="Mock response.") -> str:
            return response

        self._engine = mock

    def _check_import(self):
        pass

    def response(self, response: str):
        return self.configure(response=response)

    def parse_response(self, response):
        return response


class OpenAITextCompletion(TextCompletion):
    """
    A text completion engine that takes in a prompt and returns a completion.
    """

    COST_PER_TOKEN = {
        "text-davinci-003": 0.02 / 1000,
    }

    def __init__(
        self,
        key: Optional[str] = None,
        model: Optional[str] = "text-davinci-003",
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = 20,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ):
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
        import openai

        openai.api_key = key
        self._engine = openai.Completion.create

    def _check_import(self):
        try:
            assert openai
        except ImportError:
            raise ImportError(
                f"{self.__class__.__name__} requires the openai package."
                " Please install it with `pip install openai`."
            )

    def on_run_end(self):
        """Run after the engine has been run."""
        if self._logger is None:
            return

        self._logger.log_engine_run(
            errand_run_id=self._errand_run_id
            if hasattr(self, "_errand_run_id")
            else None,
            input=self.prompt,
            output=self.result,
            engine=f"{self.name}/{self._model}",
            cost=self.COST_PER_TOKEN[self._model]
            * self.response["usage"]["total_tokens"],
            input_tokens=self.response["usage"]["prompt_tokens"],
            output_tokens=self.response["usage"]["completion_tokens"],
        )

    def parse_response(self, response: dict) -> str:
        return response.choices[0].text

    def key(self, key: str):
        self.setup_engine(key=key)
        return self
