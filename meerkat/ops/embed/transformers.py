from typing import TYPE_CHECKING, Dict, List, Union

from meerkat.tools.lazy_loader import LazyLoader

from .encoder import Encoder

torch = LazyLoader("torch")

if TYPE_CHECKING:
    import torch


def transformers(
    variant: str = "bert-large-cased", device: Union[int, str] = "cpu"
) -> Dict[str, Encoder]:
    """Transformer encoders.

    - "text"

    Encoders will map these different modalities to the same embedding space.

    Args:
        variant (str, optional): A model name listed by `clip.available_models()`, or
            the path to a model checkpoint containing the state_dict. Defaults to
            "ViT-B/32".
        device (Union[int, str], optional): The device on which the encoders will be
            loaded. Defaults to "cpu".
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError("To embed with transformers run `pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(variant)
    model = AutoModel.from_pretrained(variant)

    model.to(device)

    def _encode(x: List[str]) -> "torch.Tensor":
        # need to coerce to list in case someone passes in a pandas series or ndarray
        x = list(x)
        return model(
            **tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(
                device=device
            )
        ).last_hidden_state[:, 0]

    return {
        "text": Encoder(
            # need to squeeze out the batch dimension for compatibility with collate
            encode=_encode,
            preprocess=lambda x: x,
        ),
    }
