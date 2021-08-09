from typing import List

from meerkat.datapanel import DataPanel
from meerkat.nn.activation import ActivationOp
from meerkat.tools.lazy_loader import LazyLoader

pl = LazyLoader("pytorch_lightning")


class ActivationCallback(pl.callbacks.Callback):
    def __init__(
        self,
        target_module: str,
        val_dp: DataPanel,
        input_columns: List[str],
        batch_size: int = 32,
    ):
        super().__init__()
        self.target_module = target_module
        self.val_dp = val_dp
        self.input_columns = input_columns
        self.batch_size = batch_size

    def on_validation_epoch_start(self, trainer, pl_module):
        self.activation_op = ActivationOp(
            pl_module.model, self.target_module, pl_module.device
        )

    def on_validation_epoch_end(self, trainer, pl_module):

        activations = self.activation_op.activation(
            dataset=self.val_dp,
            input_columns=self.input_columns,
            forward=False,
            batch_size=self.batch_size,
        )

        self.val_dp.add_column(f"activation_{trainer.current_epoch}", activations)
