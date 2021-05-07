import hashlib
import inspect
import json
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import cytoolz as tz
import progressbar
import yaml


def transpose_dict_of_lists(d: Dict):
    """Transpose a dict of lists to a list of dicts.

    Args:
        d (Dict): a dictionary mapping keys to lists

    Returns: list of dicts, each dict corresponding to a single entry
    """
    return [dict(zip(d, t)) for t in zip(*d.values())]


def convert_to_batch_column_fn(function: Callable, with_indices: bool):
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
                    batch[i],
                    indices[i],
                    *args,
                    **kwargs,
                )
            else:
                output = function(
                    batch[i],
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


def convert_to_batch_fn(function: Callable, with_indices: bool):
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
                    {k: v[i] for k, v in batch.items()},
                    indices[i],
                    *args,
                    **kwargs,
                )
            else:
                output = function(
                    {k: v[i] for k, v in batch.items()},
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


def recmerge(*objs, merge_sequences=False):
    """Recursively merge an arbitrary number of collections. For conflicting
    values, later collections to the right are given priority. By default
    (merge_sequences=False), sequences are treated as a normal value and not
    merged.

    Args:
        *objs: collections to merge
        merge_sequences: whether to merge values that are sequences

    Returns: merged collection
    """
    if isinstance(objs, tuple) and len(objs) == 1:
        # A squeeze operation since merge_with generates tuple(list_of_objs,)
        objs = objs[0]
    if all([isinstance(obj, Mapping) for obj in objs]):
        # Merges all the collections, recursively applies merging to the combined values
        return tz.merge_with(partial(recmerge, merge_sequences=merge_sequences), *objs)
    elif all([isinstance(obj, Sequence) for obj in objs]) and merge_sequences:
        # Merges sequence values by concatenation
        return list(tz.concat(objs))
    else:
        # If colls does not contain mappings, simply pick the last one
        return tz.last(objs)


def persistent_hash(s: str):
    """Compute a hash that persists across multiple Python sessions for a
    string."""
    return int(hashlib.sha224(s.encode()).hexdigest(), 16)


def strings_as_json(strings: List[str]):
    """Convert a list of strings to JSON.

    Args:
        strings: A list of str.

    Returns: JSON dump of the strings.
    """
    if len(strings) > 1:
        return json.dumps(strings)
    elif len(strings) == 1:
        return strings[0]
    else:
        return ""


def get_default_args(func) -> dict:
    """Inspect a function to get arguments that have default values.

    Args:
        func: a Python function

    Returns: dictionary where keys correspond to arguments, and values correspond to
    their defaults.
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class DownloadProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(
                maxval=total_size if total_size > 0 else 1e-2
            )
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def prettyprint(s: str) -> None:
    """Prettyprint with YAML.

    Args:
        s: string
    """
    if hasattr(s, "__dict__"):
        print(yaml.dump(s.__dict__))
    elif isinstance(s, dict):
        print(yaml.dump(s))
    else:
        print(s)


def get_all_leaf_paths(coll):
    """Returns a list of paths to all leaf nodes in a nested dict.

    Paths can travel through lists and the index is inserted into the
    path.
    """
    if isinstance(coll, Mapping):
        return list(
            tz.concat(
                map(
                    lambda t: list(map(lambda p: [t[0]] + p, get_all_leaf_paths(t[1]))),
                    coll.items(),
                )
            )
        )

    elif isinstance(coll, list):
        return list(
            tz.concat(
                map(
                    lambda t: list(map(lambda p: [t[0]] + p, get_all_leaf_paths(t[1]))),
                    enumerate(coll),
                )
            )
        )
    else:
        return [[]]


def get_all_paths(coll, prefix_path=(), stop_at=None, stop_below=None):
    """Given a collection, by default returns paths to all the leaf nodes.

    Use stop_at to truncate paths at the given key. Use stop_below to
    truncate paths one level below the given key.
    """
    assert (
        stop_at is None or stop_below is None
    ), "Only one of stop_at or stop_below can be used."
    if stop_below is not None and stop_below in str(
        tz.last(tz.take(len(prefix_path) - 1, prefix_path))
    ):
        return [[]]
    if stop_at is not None and stop_at in str(tz.last(prefix_path)):
        return [[]]
    if isinstance(coll, Mapping) or isinstance(coll, list):
        if isinstance(coll, Mapping):
            items = coll.items()
        else:
            items = enumerate(coll)

        return list(
            tz.concat(
                map(
                    lambda t: list(
                        map(
                            lambda p: [t[0]] + p,
                            get_all_paths(
                                t[1],
                                prefix_path=list(prefix_path) + [t[0]],
                                stop_at=stop_at,
                                stop_below=stop_below,
                            ),
                        )
                    ),
                    items,
                )
            )
        )
    else:
        return [[]]


def get_only_paths(coll, pred, prefix_path=(), stop_at=None, stop_below=None):
    """Get all paths that satisfy the predicate fn pred.

    First gets all paths and then filters them based on pred.
    """
    all_paths = get_all_paths(
        coll, prefix_path=prefix_path, stop_at=stop_at, stop_below=stop_below
    )
    return list(filter(pred, all_paths))


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


def nested_map(f, *args):
    """Recursively transpose a nested structure of tuples, lists, and dicts."""
    assert len(args) > 0, "Must have at least one argument."
    arg = args[0]
    if isinstance(arg, Sequence) and not isinstance(arg, str):
        return [nested_map(f, *a) for a in zip(*args)]
    elif isinstance(arg, Mapping):
        return {k: nested_map(f, *[a[k] for a in args]) for k in arg}
    else:
        return f(*args)
