from typing import Union

import torch


class DeviceMixin:
    def __init__(
        self,
        device: Union[int, str],
        *args,
        **kwargs,
    ):
        super(DeviceMixin, self).__init__(*args, **kwargs)

        if isinstance(device, str):
            if device.startswith("cuda"):
                self._cuda_device = (
                    0 if ":" not in device else int(device.split(":")[-1])
                )
            elif device.startswith("gpu"):
                self._cuda_device = (
                    0 if ":" not in device else int(device.split(":")[-1])
                )
            elif device.startswith("cpu"):
                self._cuda_device = None
        elif isinstance(device, int):
            assert device >= 0, "`device` must be >= 0."
            self._cuda_device = device
        else:
            self._cuda_device = -1
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._cuda_device == -1:
            self._device = "cuda"
        elif self._cuda_device >= 0:
            self._device = f"cuda:{self._cuda_device}"
        else:
            self._device = "cpu"

    @property
    def cuda_device(self):
        return self._device

    @property
    def device(self):
        return self._device
