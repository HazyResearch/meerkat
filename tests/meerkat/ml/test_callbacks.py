import os
import sys
from itertools import product

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch import nn
from torch.utils.data import DataLoader

from meerkat import DataPanel
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.ml import ActivationCallback, load_activations


class MockModel(pl.LightningModule):
    def __init__(self, input_size: tuple, output_size: tuple):
        super().__init__()
        self.identity = nn.Identity()
        self.linear = nn.Linear(*input_size, *output_size)

    def forward(self, x):
        return self.linear(self.identity(x))

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
            "index": np.arange(num_inputs),
            "target": np.random.randint(*output_size, size=num_inputs),
        }
    )

    return dp


def train(
    num_inputs: int = 10,
    input_size: tuple = (16,),
    output_size: tuple = (10,),
    target_module: str = "identity",
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

    model = MockModel(input_size=input_size, output_size=output_size)

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
        default_root_dir=logdir,
    )

    train_dl = DataLoader(train_dp, batch_size=batch_size, num_workers=num_workers)
    valid_dl = DataLoader(val_dp, batch_size=batch_size, num_workers=num_workers)

    trainer.fit(model, train_dl, valid_dl)

    _ = model(val_dp["input"])
    activations = activation_callback.activation_op.extractor.activation.cpu().detach()

    return model, activation_callback, activations


@pytest.mark.parametrize(
    "target_module,num_inputs,mmap,max_epochs",
    product(["identity"], [10, 20], [True, False], [1, 2]),
)
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="crashes on darwin")
def test_callback(target_module, num_inputs, mmap, max_epochs, tmpdir):
    model, act_callback, true_activations = train(
        num_inputs=num_inputs, mmap=mmap, max_epochs=max_epochs, logdir=tmpdir
    )

    for epoch in range(max_epochs):
        path = os.path.join(tmpdir, f"activations_{target_module}_{epoch}")
        if not mmap:
            path += ".mk"
        assert os.path.exists(path)

        if mmap:
            activations = {f"activation_{target_module}": open_memmap(path)}
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

        if mmap:
            assert (
                torch.from_numpy(activations[f"activation_{target_module}"].data)
                == true_activations
            ).all()

        else:
            assert (
                activations[f"activation_{target_module}"] == true_activations
            ).all()


@pytest.mark.parametrize(
    "target_module,num_inputs,mmap,max_epochs",
    product(["identity"], [10, 20], [True, False], [1, 2]),
)
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="crashes on darwin")
def test_load_activations(target_module, num_inputs, mmap, max_epochs, tmpdir):
    # TODO(Priya): Tests for non-continuous epochs
    model, act_callback, true_activations = train(
        num_inputs=num_inputs, mmap=mmap, max_epochs=max_epochs, logdir=tmpdir
    )

    activations_dp = load_activations(
        target_module=target_module,
        logdir=tmpdir,
        epochs=[*range(max_epochs)],
        mmap=mmap,
    )

    columns = [f"activation_{target_module}_{epoch}" for epoch in range(max_epochs)]
    assert set(columns) == set(activations_dp.columns)

    for col in columns:
        if mmap:
            assert (
                torch.from_numpy(activations_dp[col].data) == true_activations
            ).all()
        else:
            assert (activations_dp[col] == true_activations).all()
