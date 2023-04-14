import os
from typing import Optional

from meerkat.config import config


class OpenAIMixin:
    _organization: Optional[str] = None

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

        organization = self.organization
        if organization is None:
            if config.engines.openai_organization is not None:
                organization = config.engines.openai_organization
            elif "OPENAI_ORGANIZATION" in os.environ:
                organization = os.environ["OPENAI_ORGANIZATION"]

        openai.api_key = key
        if organization is not None:
            openai.organization = organization
        self._engine = openai.Completion()

    @property
    def organization(self):
        return self._organization
