import os
import tempfile
from itertools import product

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from meerkat import DataPanel
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.ml import ActivationCallback
from meerkat.tools.lazy_loader import LazyLoader

pl = LazyLoader("pytorch_lightning")


class MockModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        return torch.relu(self.linear(x))

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch["input"], batch["target"], batch["index"]
        loss = F.cross_entropy(self(inputs), targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch["input"], batch["target"], batch["index"]
        loss = F.cross_entropy(self(inputs), targets)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer


def generate_input(num_inputs, input_size: tuple, output_size: tuple):
    dp = DataPanel(
        {
            "input": torch.randn(num_inputs, *input_size),
            "target": np.random.randint(*output_size, size=num_inputs),
        }
    )

    return dp


def train(
    num_inputs: int = 10,
    input_size: tuple = (512,),
    output_size: tuple = (10,),
    target_module: str = "linear",
    max_epochs: int = 2,
    num_workers: int = 0,
    batch_size: int = 4,
    seed: int = 123,
    mmap: bool = True,
    logdir: str = None,
):
    pl.utilities.seed.seed_everything(seed)

    train_dp = generate_input(num_inputs, input_size, output_size)
    val_dp = generate_input(num_inputs, input_size, output_size)

    model = MockModel()

    activation_callback = ActivationCallback(
        target_module=target_module, val_len=len(val_dp), logdir=logdir, mmap=mmap
    )

    model.train()
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=max_epochs,
        log_every_n_steps=1,
        callbacks=[activation_callback],
        progress_bar_refresh_rate=None,
    )

    train_dl = DataLoader(train_dp, batch_size=batch_size, num_workers=num_workers)

    valid_dl = DataLoader(
        val_dp,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    trainer.fit(model, train_dl, valid_dl)
    return model, activation_callback


@pytest.mark.parametrize(
    "target_module,num_inputs,mmap,max_epochs",
    product(["linear"], [10, 20], [True, False], [1, 2]),
)
def test_train(target_module, num_inputs, mmap, max_epochs):
    tempdir = tempfile.TemporaryDirectory()
    model, act_callback = train(
        num_inputs=num_inputs, mmap=mmap, max_epochs=max_epochs, logdir=tempdir.name
    )

    for epoch in range(max_epochs):
        path = os.path.join(tempdir.name, f"activations_{target_module}_{epoch}")
        assert os.path.exists(path)

        # TODO(Priya): Utility functions to access these stored files/datapanels
        if mmap:
            activations = {
                f"activation_{target_module}": np.memmap(
                    path, mode="r", shape=act_callback.shape
                )
            }
            activations = DataPanel.from_batch(activations)
            assert isinstance(
                activations[f"activation_{target_module}"], NumpyArrayColumn
            )
            assert (
                activations[f"activation_{target_module}"][0].shape
                == act_callback.shape[1:]
            )

        else:
            activations = DataPanel.read(path)
            assert isinstance(activations[f"activation_{target_module}"], TensorColumn)
