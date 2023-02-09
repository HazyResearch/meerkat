"""Adapted from https://github.com/facebookresearch/detectron2."""
import importlib
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
import PIL
from tabulate import tabulate

from meerkat.tools.lazy_loader import LazyLoader
from meerkat.version import __version__

torch = LazyLoader("torch")
torchvision = LazyLoader("torchvision")

__all__ = ["collect_env_info"]


def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, "bin", "cuobjdump")
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output(
                "'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True
            )
            output = output.decode("utf-8").strip().split("\n")
            sm = []
            for line in output:
                line = re.findall(r"\.sm_[0-9]*\.", line)[0]
                sm.append(line.strip("."))
            sm = sorted(set(sm))
            return ", ".join(sm)
        else:
            return so_file + "; cannot find cuobjdump"
    except Exception:
        # unhandled failure
        return so_file


def collect_torchvision_env():
    from torch.utils.cpp_extension import CUDA_HOME

    has_cuda = torch.cuda.is_available()
    data = []
    try:
        import torchvision

        data.append(
            (
                "torchvision",
                str(torchvision.__version__)
                + " @"
                + os.path.dirname(torchvision.__file__),
            )
        )
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec("torchvision._C").origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(("torchvision arch flags", msg))
            except ImportError:
                data.append(("torchvision._C", "failed to find"))
    except (AttributeError, ModuleNotFoundError):
        data.append(("torchvision", "unknown"))
    return data


def _get_version(module_name: str, raise_error: bool = False) -> str:
    """Get version of a module from subprocess."""
    try:
        return subprocess.run(
            [module_name, "--version"], stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
    except Exception as e:
        if raise_error:
            raise e
        return "unknown"


def collect_env_info():
    has_cuda = torch.cuda.is_available()

    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("platform.platform", platform.platform()))
    data.append(("node", _get_version("node")))
    data.append(("npm", _get_version("npm")))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("meerkat", __version__))
    data.append(("numpy", np.__version__))
    data.append(("PyTorch", torch.__version__ + " @" + os.path.dirname(torch.__file__)))
    data.append(("PyTorch debug build", torch.version.debug))

    data.append(("CUDA available", has_cuda))
    if has_cuda:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))
        from torch.utils.cpp_extension import CUDA_HOME

        data.append(("CUDA_HOME", str(CUDA_HOME)))

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                nvcc = subprocess.check_output(
                    "'{}' -V | tail -n1".format(nvcc), shell=True
                )
                nvcc = nvcc.decode("utf-8").strip()
            except subprocess.SubprocessError:
                nvcc = "Not Available"
            data.append(("NVCC", nvcc))

        cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if cuda_arch_list:
            data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    data.append(("Pillow", PIL.__version__))

    # torchvision
    data.extend(collect_torchvision_env())

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass
    # Slurm info
    data.append(("SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID", "slurm not detected")))
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


if __name__ == "__main__":
    print(collect_env_info())
