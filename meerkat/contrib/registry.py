import functools
import os
from typing import Any, List, Optional, Sequence

from fvcore.common.registry import Registry as _Registry
from tabulate import tabulate

from meerkat.config import ContribOptions
from meerkat.datapanel import DataPanel


class Registry(_Registry):
    """Extension of fvcore's registry that supports aliases."""

    _ALIAS_KEYWORDS = ("_aliases", "_ALIASES")

    def __init__(self, name: str):

        super().__init__(name=name)

        self._metadata_map = {}

    def get(
        self, name: str, dataset_dir: str = None, download: bool = True, *args, **kwargs
    ) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        if dataset_dir is None:
            dataset_dir = os.path.join(ContribOptions.download_dir, name)
        return ret(dataset_dir=dataset_dir, download=download, *args, **kwargs)

    def _get_aliases(self, obj_func_or_class):
        for kw in self._ALIAS_KEYWORDS:
            if hasattr(obj_func_or_class, kw):
                return getattr(obj_func_or_class, kw)
        return []

    def register(
        self, obj: object = None, aliases: Sequence[str] = None
    ) -> Optional[object]:
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object, aliases=None) -> object:
                name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                if aliases is None:
                    aliases = self._get_aliases(func_or_class)
                if not isinstance(aliases, (list, tuple, set)):
                    aliases = [aliases]
                for alias in aliases:
                    self._do_register(alias, func_or_class)
                return func_or_class

            kwargs = {"aliases": aliases}
            if any(v is not None for v in kwargs.values()):
                return functools.partial(deco, **kwargs)
            else:
                return deco

        name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)
        if aliases is None:
            aliases = self._get_aliases(obj)
        for alias in aliases:
            self._do_register(alias, obj)

    def _do_register(self, name: str, obj: Any, **kwargs) -> None:
        self._metadata_map[name] = {"name": name, "description": obj.__doc__, **kwargs}
        return super()._do_register(name, obj)

    @property
    def names(self) -> List[str]:
        return list(self._obj_map.keys())

    @property
    def catalog(self) -> DataPanel:
        return DataPanel(data=list(self._metadata_map.values()))

    def __repr__(self) -> str:
        table = tabulate(self._metadata_map.values(), tablefmt="fancy_grid")
        return "Registry of {}:\n".format(self._name) + table


datasets = Registry("datasets")
datasets.__doc__ = """Registry for datasets in meerkat"""


@datasets.register()
def cifar10(dataset_dir: str = None, download: bool = True, **kwargs):
    """[summary]"""
    from .torchvision import get_cifar10

    return get_cifar10(download_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def imagenet(dataset_dir: str = None, download: bool = True, **kwargs):
    from .imagenet import build_imagenet_dps

    return build_imagenet_dps(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def imagenette(dataset_dir: str = None, download: bool = True, **kwargs):
    from .imagenette import build_imagenette_dp

    return build_imagenette_dp(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def celeba(dataset_dir: str = None, download: bool = True, **kwargs):
    from .celeba import get_celeba

    return get_celeba(dataset_dir=dataset_dir, download=download, **kwargs)
