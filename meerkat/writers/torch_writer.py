import torch

from meerkat.writers.abstract import AbstractWriter


class TorchWriter(AbstractWriter):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(TorchWriter, self).__init__(*args, **kwargs)

    def open(self) -> None:
        self.outputs = []

    def write(self, data, **kwargs) -> None:
        self.outputs.extend(data)

    def flush(self, *args, **kwargs):
        return torch.stack(self.outputs)

    def close(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs) -> None:
        pass
