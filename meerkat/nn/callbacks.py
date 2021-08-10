from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel
from meerkat.nn.activation import ActivationOp
from meerkat.tools.lazy_loader import LazyLoader

pl = LazyLoader("pytorch_lightning")


class ActivationCallback(pl.callbacks.Callback):
    def __init__(
        self,
        target_module: str,
        log_dir: str,  # TODO(Priya): Use trainer.log_dir?
        mmap: bool = False,
    ):
        super().__init__()
        self.target_module = target_module
        self.log_dir = log_dir
        self.mmap = mmap  # TODO(Priya): Raise Warning

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
                    shape = (len(batch), *activations.shape[1:])
                    path = f"{self.log_dir}/activation_{self.target_module}"
                    self.writer.open(str(path), shape=shape)

                else:
                    self.writer.open()
            self.writer.write(activations)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.running_sanity_check:
            activations = {f"activation_{self.target_module}": self.writer.flush()}
            activations = DataPanel.from_batch(activations)

        # TODO(Priya): Store the datapanels, write to disk, or log
