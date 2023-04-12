import os
from typing import Optional

from meerkat.config import config


class OpenAIMixin:
    def authenticate(self, key: Optional[str] = None):
        import openai

        if key is None:
            if config.engines.openai_api_key is not None:
                key = config.engines.openai_api_key
            elif "OPENAI_API_KEY" in os.environ:
                key = os.environ["OPENAI_API_KEY"]
            else:
                raise ValueError(
                    "No OpenAI API key was provided. Please provide one using the `key`"
                    " argument or by setting the `engines.openai_api_key` "
                    " configuration variable."
                )

        openai.api_key = key
        self._engine = openai.Completion()
