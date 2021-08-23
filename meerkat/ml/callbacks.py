import warnings

from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel
from meerkat.ml.activation import ActivationOp
from meerkat.tools.lazy_loader import LazyLoader

pl = LazyLoader("pytorch_lightning")


class ActivationCallback(pl.callbacks.Callback):
    def __init__(
        self,
        target_module: str,
        val_len: int,
        logdir: str,  # TODO(Priya): Use trainer.log_dir?
        mmap: bool = False,
    ):
        super().__init__()
        self.target_module = target_module
        self.val_len = val_len
        self.logdir = logdir
        self.mmap = mmap

        if self.mmap:
            warnings.warn(
                "Activations will be stored as numpy array when using memmapping."
            )

    def on_validation_epoch_start(self, trainer, pl_module):
        if not trainer.running_sanity_check:
            self.activation_op = ActivationOp(
                pl_module.model, self.target_module, pl_module.device
            )
            self.writer = TensorColumn.get_writer(mmap=self.mmap)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not trainer.running_sanity_check:
            activations = self.activation_op.extractor.activation.cpu().detach()

            # Use the first batch for setup
            if batch_idx == 0:
                if self.mmap:
                    shape = (self.val_len, *activations.shape[1:])

                    # TODO(Priya): File name format
                    file = f"activations_{self.target_module}_{trainer.current_epoch}"
                    path = f"{self.logdir}/{file}"
                    self.writer.open(str(path), shape=shape)

                else:
                    self.writer.open()
            self.writer.write(activations)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.running_sanity_check:
            activations = {f"activation_{self.target_module}": self.writer.flush()}
            activations = DataPanel.from_batch(activations)

            if not self.mmap:
                file = f"activations_{self.target_module}_{trainer.current_epoch}"
                path = f"{self.logdir}/{file}"
                activations.write(path)
