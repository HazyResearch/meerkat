"""A chatbot to interact with a large language model."""
from typing import Generator, List, Optional, Tuple
from pydantic import BaseModel, validator
from meerkat.tools.lazy_loader import LazyLoader


openai = LazyLoader("openai")


class Message(BaseModel):
    role: str
    content: str

    @validator("role")
    def role_must_be_system_or_user(cls, v):
        if v not in ["system", "user"]:
            raise ValueError("role must be system or user")
        return v

def chat(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    history: List[Message] = [],
    system_prompt: str = "You are a helpful assistant.",
) -> Tuple[Optional[str], dict]:
    """Chat with a large language model."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        + [message.dict() for message in history]
        + [
            {"role": "user", "content": prompt},
        ],
    )
    try:
        return response['choices'][0]['message']['content'], response
    except KeyError:
        return None, response


def chat_streaming(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    history: List[Message] = [],
    system_prompt: str = "You are a helpful assistant.",
) -> Generator[Optional[str], None, None]:
    """Chat with a large language model."""
    for response in openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        + [message.dict() for message in history]
        + [
            {"role": "user", "content": prompt},
        ],
    ):
        try:
            yield response["choices"][0]["delta"]["content"]
        except KeyError:
            yield None
