import inspect
import sys
import types
import typing
import warnings
import weakref
from collections import defaultdict
from collections.abc import Mapping
from functools import reduce, wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import dill
import numpy as np
import pandas as pd
import yaml
from yaml.constructor import ConstructorError

from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")


def is_subclass(v, cls):
    """Check if `v` is a subclass of `cls`, with guard for TypeError."""
    try:
        _is_subclass = issubclass(v, cls)
    except TypeError:
        _is_subclass = False

    return _is_subclass


def has_var_kwargs(fn: Callable) -> bool:
    """Check if a function has variable keyword arguments e.g. **kwargs.

    Args:
        fn: The function to check.

    Returns:
        True if the function has variable keyword arguments, False otherwise.
    """
    sig = inspect.signature(fn)
    params = sig.parameters.values()
    return any([True for p in params if p.kind == p.VAR_KEYWORD])


def has_var_args(fn: Callable) -> bool:
    """Check if a function has variable positional arguments e.g. *args.

    Args:
        fn: The function to check.

    Returns:
        True if the function has variable positional arguments, False otherwise.
    """
    sig = inspect.signature(fn)
    params = sig.parameters.values()
    return any([True for p in params if p.kind == p.VAR_POSITIONAL])


def get_type_hint_args(type_hint):
    """Get the arguments of a type hint."""
    if sys.version_info >= (3, 8):
        # Python > 3.8
        return typing.get_args(type_hint)
    else:
        return type_hint.__args__


def get_type_hint_origin(type_hint):
    """Get the origin of a type hint."""
    if sys.version_info >= (3, 8):
        # Python > 3.8
        return typing.get_origin(type_hint)
    else:
        return type_hint.__origin__


class classproperty(property):
    """Taken from https://stackoverflow.com/a/13624858.

    The behavior of class properties using the @classmethod and
    @property decorators has changed across Python versions. This class
    (should) provide consistent behavior across Python versions. See
    https://stackoverflow.com/a/1800999 for more information.
    """

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def deprecated(replacement: Optional[str] = None):
    """This is a decorator which can be used to mark functions as deprecated.

    It will result in a warning being emitted when the function is used.
    """

    def _decorator(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            warnings.warn(
                "Call to deprecated function {}.".format(func.__name__) + ""
                if new_func is None
                else " Use {} instead.".format(replacement),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return new_func

    return _decorator


class WeakMapping(Mapping):
    def __init__(self):
        self.refs: Dict[Any, weakref.ReferenceType] = {}

    def __getitem__(self, key: str):
        ref = self.refs[key]
        obj = ref()
        if obj is None:
            raise KeyError(f"Object with key {key} no longer exists")
        return obj

    def __setitem__(self, key: str, value: Any):
        self.refs[key] = weakref.ref(value)

    def __delitem__(self, key: str):
        del self.refs[key]

    def __iter__(self):
        return iter(self.refs)

    def __len__(self):
        return len(self.refs)


def nested_getattr(obj, attr, *args):
    """Get a nested property from an object.

    # noqa: E501
    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    return reduce(lambda o, a: getattr(o, a, *args), [obj] + attr.split("."))


def nested_apply(obj: object, fn: callable, base_types: Tuple[type] = ()):
    if isinstance(obj, base_types):
        return fn(obj)
    elif isinstance(obj, list):
        return [nested_apply(v, fn=fn, base_types=base_types) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(nested_apply(v, fn=fn, base_types=base_types) for v in obj)
    elif isinstance(obj, dict):
        return {
            k: nested_apply(v, fn=fn, base_types=base_types) for k, v in obj.items()
        }
    else:
        return fn(obj)


BACKWARDS_COMPAT_REPLACEMENTS = [
    ("meerkat.ml", "meerkat.nn"),
    ("meerkat.columns.numpy_column", "meerkat.columns.tensor.numpy"),
    ("NumpyArrayColumn", "NumPyTensorColumn"),
    ("meerkat.columns.tensor_column", "meerkat.columns.tensor.torch"),
    ("meerkat.columns.pandas_column", "meerkat.columns.scalar.pandas"),
    ("meerkat.columns.arrow_column", "meerkat.columns.scalar.arrow"),
    ("meerkat.columns.image_column", "meerkat.columns.deferred.image"),
    ("meerkat.columns.file_column", "meerkat.columns.deferred.file"),
    ("meerkat.columns.list_column", "meerkat.columns.object.base"),
    ("meerkat.block.lambda_block", "meerkat.block.deferred_block"),
    (
        "meerkat.interactive.app.src.lib.component.filter",
        "meerkat.interactive.app.src.lib.component.core.filter",
    ),
    ("ListColumn", "ObjectColumn"),
    ("LambdaBlock", "DeferredBlock"),
    ("NumpyBlock", "NumPyBlock"),
]


class MeerkatDumper(yaml.Dumper):
    @staticmethod
    def _pickled_object_representer(dumper, data):
        return dumper.represent_mapping(
            "!PickledObject", {"class": data.__class__, "pickle": dill.dumps(data)}
        )

    @staticmethod
    def _function_representer(dumper, data):
        if data.__name__ == "<lambda>":
            return dumper.represent_mapping(
                "!Lambda",
                {"code": inspect.getsource(data), "pickle": dill.dumps(data)},
            )

        if "<locals>" in data.__qualname__:
            return dumper.represent_mapping(
                "!NestedFunction",
                {"code": inspect.getsource(data), "pickle": dill.dumps(data)},
            )

        return dumper.represent_name(data)


MeerkatDumper.add_multi_representer(object, MeerkatDumper._pickled_object_representer)
MeerkatDumper.add_representer(types.FunctionType, MeerkatDumper._function_representer)


class MeerkatLoader(yaml.FullLoader):
    """PyYaml does not load unimported modules for safety reasons.

    We want to allow importing only meerkat modules
    """

    def find_python_module(self, name: str, mark, unsafe=False):
        try:
            return super().find_python_module(name=name, mark=mark, unsafe=unsafe)
        except ConstructorError as e:
            if name.startswith("meerkat."):
                __import__(name)
            else:
                raise e
            return super().find_python_module(name=name, mark=mark, unsafe=unsafe)

    def find_python_name(self, name: str, mark, unsafe=False):
        for old, new in BACKWARDS_COMPAT_REPLACEMENTS:
            if old in name:
                name = name.replace(old, new)

        if "." in name:
            module_name, _ = name.rsplit(".", 1)
        else:
            module_name = "builtins"

        try:
            return super().find_python_name(name=name, mark=mark, unsafe=unsafe)
        except ConstructorError as e:
            if name.startswith("meerkat."):
                __import__(module_name)
            else:
                raise e
            return super().find_python_name(name=name, mark=mark, unsafe=unsafe)

    @staticmethod
    def _pickled_object_constructor(loader, node):
        data = loader.construct_mapping(node)
        return dill.loads(data["pickle"])

    @staticmethod
    def _function_constructor(loader, node):
        data = loader.construct_mapping(node)
        return dill.loads(data["pickle"])


MeerkatLoader.add_constructor(
    "!PickledObject", MeerkatLoader._pickled_object_constructor
)
MeerkatLoader.add_constructor("!Lambda", MeerkatLoader._function_constructor)


def dump_yaml(obj: Any, path: str, **kwargs):
    with open(path, "w") as f:
        yaml.dump(obj, f, Dumper=MeerkatDumper, **kwargs)


def load_yaml(path: str, **kwargs):
    with open(path, "r") as f:
        return yaml.load(f, Loader=MeerkatLoader, **kwargs)


class MeerkatUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            for old, new in BACKWARDS_COMPAT_REPLACEMENTS:
                if old in module:
                    module = module.replace(old, new)
            return super().find_class(module, name)


def meerkat_dill_load(path: str):
    """Load dill file with backwards compatibility for old column names."""
    return MeerkatUnpickler(open(path, "rb")).load()


def convert_to_batch_column_fn(
    function: Callable, with_indices: bool, materialize: bool = True, **kwargs
):
    """Batch a function that applies to an example."""

    def _function(batch: Sequence, indices: Optional[List[int]], *args, **kwargs):
        # Pull out the batch size
        batch_size = len(batch)

        # Iterate and apply the function
        outputs = None
        for i in range(batch_size):

            # Apply the unbatched function
            if with_indices:
                output = function(
                    batch[i] if materialize else batch[i],
                    indices[i],
                    *args,
                    **kwargs,
                )
            else:
                output = function(
                    batch[i] if materialize else batch[i],
                    *args,
                    **kwargs,
                )

            if i == 0:
                # Create an empty dict or list for the outputs
                outputs = defaultdict(list) if isinstance(output, dict) else []

            # Append the output
            if isinstance(output, dict):
                for k in output.keys():
                    outputs[k].append(output[k])
            else:
                outputs.append(output)

        if isinstance(outputs, dict):
            return dict(outputs)
        return outputs

    if with_indices:
        # Just return the function as is
        return _function
    else:
        # Wrap in a lambda to apply the indices argument
        return lambda batch, *args, **kwargs: _function(batch, None, *args, **kwargs)


def convert_to_batch_fn(
    function: Callable, with_indices: bool, materialize: bool = True, **kwargs
):
    """Batch a function that applies to an example."""

    def _function(
        batch: Dict[str, List], indices: Optional[List[int]], *args, **kwargs
    ):
        # Pull out the batch size
        batch_size = len(batch[list(batch.keys())[0]])

        # Iterate and apply the function
        outputs = None
        for i in range(batch_size):

            # Apply the unbatched function
            if with_indices:
                output = function(
                    {k: v[i] if materialize else v[i] for k, v in batch.items()},
                    indices[i],
                    *args,
                    **kwargs,
                )
            else:
                output = function(
                    {k: v[i] if materialize else v[i] for k, v in batch.items()},
                    *args,
                    **kwargs,
                )

            if i == 0:
                # Create an empty dict or list for the outputs
                outputs = defaultdict(list) if isinstance(output, dict) else []

            # Append the output
            if isinstance(output, dict):
                for k in output.keys():
                    outputs[k].append(output[k])
            else:
                outputs.append(output)

        if isinstance(outputs, dict):
            return dict(outputs)
        return outputs

    if with_indices:
        # Just return the function as is
        return _function
    else:
        # Wrap in a lambda to apply the indices argument
        return lambda batch, *args, **kwargs: _function(batch, None, *args, **kwargs)


def convert_to_python(obj: Any):
    """Utility for converting NumPy and torch dtypes to native python types.

    Useful when sending objects to frontend.
    """
    import torch

    if torch.is_tensor(obj):
        obj = obj.numpy()

    if isinstance(obj, np.generic):
        obj = obj.item()

    return obj


def translate_index(index, length: int):
    def _is_batch_index(index):
        # np.ndarray indexed with a tuple of length 1 does not return an np.ndarray
        # so we match this behavior
        return not (
            isinstance(index, int) or (isinstance(index, tuple) and len(index) == 1)
        )

    # `index` should return a single element
    if not _is_batch_index(index):
        return index

    from ..columns.scalar.abstract import ScalarColumn
    from ..columns.tensor.abstract import TensorColumn

    if isinstance(index, pd.Series):
        index = index.values

    if torch.is_tensor(index):
        index = index.numpy()

    if isinstance(index, tuple) or isinstance(index, list):
        index = np.array(index)

    if isinstance(index, ScalarColumn):
        index = index.to_numpy()

    if isinstance(index, TensorColumn):
        if len(index.shape) == 1:
            index = index.to_numpy()
        else:
            raise TypeError(
                "`TensorColumn` index must have 1 axis, not {}".format(len(index.shape))
            )

    # `index` should return a batch
    if isinstance(index, slice):
        # int or slice index => standard list slicing
        indices = np.arange(*index.indices(length))

    elif isinstance(index, np.ndarray):
        if len(index.shape) != 1:
            raise TypeError(
                "`np.ndarray` index must have 1 axis, not {}".format(len(index.shape))
            )
        if index.dtype == bool:
            indices = np.where(index)[0]
        else:
            return index
    else:
        raise TypeError("Object of type {} is not a valid index".format(type(index)))
    return indices


def choose_device(device: str = "auto"):
    """Choose the device to use for a Meerkat operation."""
    from meerkat.config import config

    if not config.system.use_gpu:
        return "cpu"

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    return device
