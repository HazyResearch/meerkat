"""Identifiers for objects in Meerkat."""
from __future__ import annotations

import ast
import json
from typing import Any, Callable, List, Union


class Identifier:
    """Class for creating identifiers for objects in Robustness Gym."""

    def __init__(
        self,
        _name: str,
        _index: Union[str, int] = None,
        **kwargs,
    ):

        self._name = _name
        self._index = str(_index) if _index is not None else None
        self._parameters = kwargs

        # Add the parameter
        for param, value in self.parameters.items():
            self.add_parameter(param, value)

    @property
    def name(self):
        """Base name."""
        return self._name

    @property
    def index(self):
        """Index associated with the identifier."""
        return self._index

    @property
    def parameters(self):
        """Additional parameters contained in the identifier."""
        return self._parameters

    @classmethod
    def range(cls, n: int, _name: str, **kwargs) -> List[Identifier]:
        """Create a list of identifiers, with index varying from 1 to `n`."""

        if n > 1:
            return [cls(_name=_name, _index=i, **kwargs) for i in range(1, n + 1)]
        return [cls(_name=_name, **kwargs)]

    def __call__(self, **kwargs):
        """Call the identifier with additional parameters to return a new
        identifier."""
        ident = Identifier.loads(self.dumps())
        for parameter, value in kwargs.items():
            ident.add_parameter(parameter, value)
        return ident

    def __repr__(self):
        params = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        if self.index is not None:
            return (
                f"{self.name}-{self.index}({params})"
                if len(params) > 0
                else f"{self.name}-{self.index}"
            )
        return f"{self.name}({params})" if len(params) > 0 else f"{self.name}"

    def __hash__(self):
        # return persistent_hash(str(self))
        return hash(str(self))

    def __eq__(self, other: Union[Identifier, str]):
        return str(self) == str(other)

    def dumps(self):
        """Dump the identifier to JSON."""
        return json.dumps(self.__dict__)

    @staticmethod
    def _parse_args(s: str):
        """https://stackoverflow.com/questions/49723047/parsing-a-string-as-a-
        python-argument-list."""
        args = "f({})".format(s)
        tree = ast.parse(args)
        funccall = tree.body[0].value
        # return {arg.arg: ast.literal_eval(arg.value) for arg in funccall.keywords}
        params = {}
        for arg in funccall.keywords:
            try:
                params[arg.arg] = ast.literal_eval(arg.value)
            except ValueError:
                params[arg.arg] = arg.value.id
        return params

    @classmethod
    def parse(cls, s: str) -> Identifier:
        """Parse in an identifier from string."""
        # Parse out the various components
        if "(" in s:
            name_index, params = s.split("(")
            params = params.split(")")[0]
        else:
            name_index = s
            params = None

        # Create the name and index
        if "-" in name_index:
            name, index = name_index.split("-")[:-1], name_index.split("-")[-1]
            name = "-".join(name)
            if index.isnumeric():
                index = int(index)
            else:
                name = "-".join([name, index])
                index = None
        else:
            name = name_index
            index = None

        # Parse out the params
        if params is not None:
            params = cls._parse_args(params)
        else:
            params = {}

        return cls(_name=name, _index=index, **params)

    def without(self, *params) -> Identifier:
        """Returns an identifier without `params`."""
        return Identifier(
            self.name,
            self.index,
            **{k: v for k, v in self.parameters.items() if k not in set(params)},
        )

    @classmethod
    def loads(cls, s: str):
        """Load the identifier from JSON."""
        identifier = Identifier(_name="")
        identifier.__dict__ = json.loads(s)
        return identifier

    def add_parameter(self, parameter: str, value: Any) -> None:
        """Add a parameter to the identifier."""
        if isinstance(value, Callable):
            self.parameters[parameter] = ".".join(
                [str(value.__module__), str(value.__name__)]
            )
        else:
            self.parameters[parameter] = value


# Assign Id as an alias for the Identifier class
Id = Identifier
