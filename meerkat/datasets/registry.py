import functools
from typing import Any, List, Optional, Sequence

from fvcore.common.registry import Registry as _Registry
from tabulate import tabulate

from meerkat.dataframe import DataFrame


class Registry(_Registry):
    """Extension of fvcore's registry that supports aliases."""

    _ALIAS_KEYWORDS = ("_aliases", "_ALIASES")

    def __init__(self, name: str):
        super().__init__(name=name)

        self._metadata_map = {}

    def get(self, name: str, **kwargs) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )

        return ret(**kwargs)()

    def get_obj(self, name: str) -> type:
        return self._obj_map[name]

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
    def catalog(self) -> DataFrame:
        rows = []
        for name, builder in self:
            rows.append(builder.info.__dict__)
        return DataFrame(rows)

    def __repr__(self) -> str:
        table = tabulate(self._metadata_map.values(), tablefmt="fancy_grid")
        return "Registry of {}:\n".format(self._name) + table


datasets = Registry("datasets")
datasets.__doc__ = """Registry for datasets in meerkat"""
