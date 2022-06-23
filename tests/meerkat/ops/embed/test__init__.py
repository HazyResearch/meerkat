import hashlib
import os

import numpy as np
import PIL
import pytest
import torch
from PIL import Image

import meerkat as mk
from meerkat import embed
from meerkat.ops.embed import encoders
from meerkat.ops.embed.encoder import Encoder


class ImageColumnTestBed:
    def __init__(
        self,
        tmpdir: str,
        length: int = 16,
    ):
        self.image_paths = []
        self.image_arrays = []
        self.ims = []
        self.data = []

        for i in range(0, length):
            self.image_arrays.append((i * np.ones((4, 4, 3))).astype(np.uint8))
            im = Image.fromarray(self.image_arrays[-1])
            self.ims.append(im)
            self.data.append(im)
            filename = "{}.png".format(i)
            im.save(os.path.join(tmpdir, filename))
            self.image_paths.append(os.path.join(tmpdir, filename))

        self.col = mk.ImageColumn.from_filepaths(
            self.image_paths,
            loader=Image.open,
        )


class TextColumnTestBed:
    def __init__(self, length: int = 16):
        self.data = ["Row " * idx for idx in range(length)]
        self.col = mk.PandasSeriesColumn(self.data)


EMB_SIZE = 4


def simple_encode(batch: torch.Tensor):
    value = batch.to(torch.float32).mean(dim=-1, keepdim=True)
    return torch.ones(batch.shape[0], EMB_SIZE) * value


def simple_image_transform(image: PIL.Image):
    return torch.tensor(np.asarray(image)).to(torch.float32)


def simple_text_transform(text: str):
    return torch.tensor(
        [
            int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest(), "big") % 100
            for token in text.split(" ")
        ]
    )[:1]


def _simple_encoder(variant: str = "ViT-B/32", device: str = "cpu"):
    return {
        "image": Encoder(encode=simple_encode, preprocess=simple_image_transform),
        "text": Encoder(encode=simple_encode, preprocess=simple_text_transform),
    }


@pytest.fixture()
def simple_encoder(monkeypatch):
    if "_simple_encoder" not in encoders.names:
        encoders.register(_simple_encoder)
    return _simple_encoder


def test_embed_images(tmpdir: str, simple_encoder):
    image_testbed = ImageColumnTestBed(tmpdir=tmpdir)

    dp = mk.DataPanel({"image": image_testbed.col})
    dp = embed(
        data=dp,
        input="image",
        encoder="_simple_encoder",
        batch_size=4,
        num_workers=0,
    )

    assert isinstance(dp, mk.DataPanel)
    assert "_simple_encoder(image)" in dp
    assert (
        simple_image_transform(dp["image"][0]).mean()
        == dp["_simple_encoder(image)"][0].mean()
    )


def test_embed_text(simple_encoder):
    testbed = TextColumnTestBed()

    dp = mk.DataPanel({"text": testbed.col})
    dp = embed(
        data=dp,
        input="text",
        encoder="_simple_encoder",
        batch_size=4,
        num_workers=0,
    )

    assert isinstance(dp, mk.DataPanel)
    assert "_simple_encoder(text)" in dp
    assert (
        simple_text_transform(dp["text"][0]).to(torch.float32).mean()
        == dp["_simple_encoder(text)"][0].mean()
    )
