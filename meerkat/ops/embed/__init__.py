import os
from typing import Callable, Union

import PIL
import torch

import meerkat as mk
from meerkat.tools.utils import choose_device

from .bit import bit
from .clip import clip
from .encoder import Encoder
from .registry import encoders
from .robust import robust
from .transformers import transformers

__all__ = ["clip", "bit", "transformers", "robust", "embed"]


def infer_modality(col: mk.AbstractColumn):

    if isinstance(col, mk.ImageColumn):
        return "image"
    elif isinstance(col, (mk.PandasSeriesColumn, str)):
        return "text"
    else:
        raise ValueError(f"Cannot infer modality from column of type {type(col)}.")


# @cache(params=["encoder", "modality", ""])
def embed(
    data: Union[mk.DataPanel, mk.AbstractColumn, str, PIL.Image.Image],
    input: str = None,
    encoder: Union[str, Encoder] = "clip",
    modality: str = None,
    out_col: str = None,
    device: Union[int, str] = "auto",
    mmap_dir: str = None,
    num_workers: int = 0,
    batch_size: int = 128,
    **kwargs,
) -> Union[mk.DataPanel, mk.AbstractColumn]:
    """Embed a column of data with an encoder from the encoder registry.

    Examples
    --------
    Suppose you have an Image dataset (e.g. Imagenette, CIFAR-10) loaded into a
    `Meerkat DataPanel <https://github.com/robustness-gym/meerkat>`_. You can embed the
    images in the dataset with CLIP using a code snippet like:

    .. code-block:: python

        import meerkat as mk

        dp = mk.datasets.get("imagenette")

        dp = mk.embed(
            data=dp,
            input_col="img",
            encoder="clip"
        )


    Args:
        data (Union[mk.DataPanel, mk.AbstractColumn]): A datapanel or column
            containing the data to embed.
        input_col (str, optional): If ``data`` is a datapanel, the name of the column
            to embed. If ``data`` is a column, then the parameter is ignored. Defaults
            to None.
        encoder (Union[str, Encoder], optional): Name of the encoder to use. List
            supported encoders with ``domino.encoders``. Defaults to "clip".
            Alternatively, pass an :class:`~domino._embed.encoder.Encoder` object
            containing a custom encoder.
        modality (str, optional): The modality of the data to be embedded. Defaults to
            None, in which case the modality is inferred from the type of the input
            column.
        out_col (str, optional): The name of the column where the embeddings are stored.
            Defaults to None, in which case it is ``"{encoder}({input_col})"``.
        device (Union[int, str], optional): The device on which. Defaults to "cpu".
        mmap_dir (str, optional): The path to directory where a memory-mapped file
            containing the embeddings will be written. Defaults to None, in which case
            the embeddings are not memmapped.
        num_workers (int, optional): Number of worker processes used to load the data
            from disk. Defaults to 4.
        batch_size (int, optional): Size of the batches to  used . Defaults to 128.
        **kwargs: Additional keyword arguments are passed to the encoder. To see
            supported arguments for each encoder, see the encoder documentation (e.g.
            :func:`~domino._embed.clip`).

    Returns:
        mk.DataPanel: A view of ``data`` with a new column containing the embeddings.
        This column will be named according to the ``out_col`` parameter.
    """
    col = data if isinstance(data, mk.AbstractColumn) else data[input]

    device = choose_device(device)

    if out_col is None:
        out_col = f"{encoder}({input})"

    if modality is None:

        modality = infer_modality(col=col)

    encoder = encoders.get(encoder, device=device, **kwargs)

    if modality not in encoder:
        raise ValueError(f'Encoder "{encoder}" does not support modality "{modality}".')

    encoder = encoder[modality]

    out = _embed(
        col=col,
        encode=encoder.encode,
        preprocess=encoder.preprocess,
        collate=encoder.collate,
        device=device,
        mmap_dir=mmap_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    if isinstance(data, mk.DataPanel):
        data[out_col] = out
        return data
    else:
        return out


def _embed(
    col: mk.AbstractColumn,
    encode: Callable,
    preprocess: Callable,
    collate: Callable,
    device: int = None,
    mmap_dir: str = None,
    num_workers: int = 0,
    batch_size: int = 128,
):
    def _encode(x):
        return encode(_prepare_input(x)).cpu().detach().numpy()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if preprocess is not None:
        embed_input = col.to_lambda(preprocess)
    else:
        embed_input = col

    if collate is not None:
        embed_input.collate_fn = collate

    def _prepare_input(x):
        if isinstance(x, mk.AbstractColumn):
            x = x.data
        if torch.is_tensor(x):
            x = x.to(device)
        return x

    with torch.no_grad():
        out = embed_input.map(
            _encode,
            pbar=True,
            is_batched_fn=True,
            batch_size=batch_size,
            num_workers=num_workers,
            mmap=mmap_dir is not None,
            mmap_path=None
            if mmap_dir is None
            else os.path.join(mmap_dir, "emb_mmap.npy"),
            flush_size=128,
        )
    return out
