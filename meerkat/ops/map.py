import warnings
from inspect import signature
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Sequence, Tuple, Type, Union

import meerkat.tools.docs as docs
from meerkat.block.abstract import BlockView

if TYPE_CHECKING:
    from meerkat.columns.abstract import Column
    from meerkat.columns.deferred.base import DeferredColumn
    from meerkat.dataframe import DataFrame


_SHARED_DOCS_ = {
    "input_description": docs.DescriptionSection(
        """
    *What gets passed to function?*

    *   If ${data} is a :class:`DataFrame` and ``outputs`` is not passed, then the \
        function's signature is inspected to determine which columns to pass as \
        keyword arguments to the function.
        For example, if the function is
        ``lambda age, residence: age > 18 and residence == "NY"``, then
        the columns ``age`` and ``residence`` will be passed to the function. If the
        columns are not present in the DataFrame, then a `ValueError` will be raised.
    *   If ${data} is a :class:`DataFrame` and ``outputs`` is  ``"single"``, then \
        mapping between columns and function arguments can be overridden by passing
        a the ``inputs`` argument.
    *   If ${data} is a :class:`Column` then values of the
        column are passed as a single positional argument to the function. The
        ``inputs`` argument is ignored.
    """
    ),
    "function": docs.Arg(
        """
        function (Callable): The function that will be applied to the rows of
            ``${data}``.
        """
    ),
    "is_batched_fn": docs.Arg(
        """
        is_batched_fn (bool, optional): Whether the function must be applied on a
            batch of rows. Defaults to False.
        """
    ),
    "batch_size": docs.Arg(
        """
        batch_size (int, optional): The size of the batch. Defaults to 1.
        """
    ),
    "inputs": docs.Arg(
        """
        inputs (Dict[str, str], optional): Dictionary mapping column names in
            ``${data}`` to keyword arguments of ``function``. Ignored if ``${data}`` is
            a column. When calling ``function`` values from the columns will be fed to
            the corresponding keyword arguments. Defaults to None, in which case it
            inspects the signature of the function. It then finds the columns with the
            same names in the DataFrame and passes the corresponding values to the
            function. If the function takes a non-default argument that is not a
            column in the DataFrame, the operation will raise a `ValueError`.
        """
    ),
    "outputs": docs.Arg(
        """
        outputs (Union[Dict[any, str], Tuple[str]], optional): Controls how the output
            of ``function`` is mapped to the output of :func:`${name}`.
            Defaults to ``None``.

            *   If ``None``: the output is inferred from the return type of the
                function. See explanation above.
            *   If ``"single"``: a single :class:`DeferredColumn` is returned.
            *   If a ``Dict[any, str]``: then a :class:`DataFrame` containing
                DeferredColumns is returned. This is useful when the output of
                ``function`` is a ``Dict``. ``outputs`` maps the outputs of ``function``
                to column names in the resulting :class:`DataFrame`.
            *   If a ``Tuple[str]``: then a :class:`DataFrame` containing
                output :class:`DeferredColumn` is returned. This is useful when the
                of ``function`` is a ``Tuple``. ``outputs`` maps the outputs of
                ``function`` to column names in the resulting :class:`DataFrame`.
        """
    ),
    "output_type": docs.Arg(
        """
        output_type (Union[Dict[str, type], type], optional): Coerce the column.
            Defaults to None.
        """
    ),
    "materialize": docs.Arg(
        """
        materialize (bool, optional): Whether to materialize the input column(s).
            Defaults to True.
        """
    ),
    "use_ray": docs.Arg(
        """
        use_ray (bool): Use Ray to parallelize the computation. Defaults to False.
        """
    ),
    "num_blocks": docs.Arg(
        """
        num_blocks (int): When using Ray, the number of blocks to split the data into.
            Defaults to 100.
        """
    ),
    "blocks_per_window": docs.Arg(
        """
        blocks_per_window (int): When using Ray, the number of blocks to process in a
            single Ray task. Defaults to 10.
        """
    ),
    "pbar": docs.Arg(
        """
        pbar (bool): Show a progress bar. Defaults to False.
        """
    ),
}


@docs.doc(source=_SHARED_DOCS_, data="data", name="defer")
def defer(
    data: Union["DataFrame", "Column"],
    function: Callable,
    is_batched_fn: bool = False,
    batch_size: int = 1,
    inputs: Union[Mapping[str, str], Sequence[str]] = None,
    outputs: Union[Mapping[any, str], Sequence[str]] = None,
    output_type: Union[Mapping[str, Type["Column"]], Type["Column"]] = None,
    materialize: bool = True,
) -> Union["DataFrame", "DeferredColumn"]:
    """Create one or more DeferredColumns that lazily applies a function to
    each row in ${data}.

    This function shares nearly the exact same signature
    with :func:`map`, the difference is that :func:`~meerkat.defer` returns a column
    that has not yet been computed. It is a placeholder for a column that will be
    computed later.

    Learn more in the user guide: :ref:`guide/dataframe/ops/mapping/deferred`.

    {input_description}

    *What gets returned by defer?*

    *   If ``function`` returns a single value, then ``defer``
        will return a :class:`DeferredColumn` object.

    *   If ``function`` returns a dictionary, then ``defer`` will return a
        :class:`DataFrame` containing :class:`DeferredColumn` objects. The keys of the
        dictionary are used as column names. The ``outputs`` argument can be used to
        override the column names.

    *   If ``function`` returns a tuple, then ``defer`` will return a :class:`DataFrame`
        containing :class:`DeferredColumn` objects. The column names will be integers.
        The column names can be overriden by passing a tuple to the ``outputs``
        argument.

    *   If ``function`` returns a tuple or a dictionary, then passing ``"single"`` to
        the ``outputs`` argument will cause ``defer`` to return a single
        :class:`DeferredColumn` that materializes to a :class:`ObjectColumn`.

    *How do you execute the deferred map?*

    Depending on ``function`` and the ``outputs`` argument, returns either a
    :class:`DeferredColumn` or a :class:`DataFrame`. Both are **callables**. To execute
    the deferred map, simply call the returned object.

    .. note::
        This function is also available as a method of :class:`DataFrame` and
        :class:`Column` under the name ``defer``.


    Args:
        ${data} (DataFrame): The :class:`DataFrame` or :class:`Column` to which the
            function will be applied.
        ${function}
        ${is_batched_fn}
        ${batch_size}
        ${inputs}
        ${outputs}
        ${output_type}
        ${materialize}

    Returns:
        Union[DataFrame, DeferredColumn]: A :class:`DeferredColumn` or a
            :class:`DataFrame` containing :class:`DeferredColumn` representing the
            deferred map.

    Examples
    ---------
    We start with a small DataFrame of voters with two columns: `birth_year`, which
    contains the birth year of each person, and `residence`, which contains the state in
    which each person lives.

    .. ipython:: python

        import datetime
        import meerkat as mk

        df = mk.DataFrame({
            "birth_year": [1967, 1993, 2010, 1985, 2007, 1990, 1943],
            "residence": ["MA", "LA", "NY", "NY", "MA", "MA", "LA"]
        })


    **Single input column.** Lazily create a column of birth years to a column of ages.

    .. ipython:: python

        df["age"] = df["birth_year"].defer(
            lambda x: datetime.datetime.now().year - x
        )
        df["age"]

    We can materialize the deferred map (*i.e.* run it) by calling the column.

    .. ipython:: python

        df["age"]()


    **Multiple input columns.** Lazily create a column of birth years to a column of
    ages.

    .. ipython:: python

        df["ma_eligible"] = df.defer(
            lambda age, residence: (residence == "MA") and (age >= 18)
        )
        df["ma_eligible"]()
    """
    from meerkat import DeferredColumn
    from meerkat.block.deferred_block import DeferredBlock, DeferredOp
    from meerkat.columns.abstract import Column, infer_column_type
    from meerkat.dataframe import DataFrame

    base_function = function
    # prepare arguments for LambdaOp
    if isinstance(data, Column):
        args = [data]
        kwargs = {}
    elif isinstance(data, DataFrame):
        args, kwargs = None, None
        if inputs == "row":
            pass
        elif isinstance(inputs, Mapping):
            args = []
            kwargs = {kw: data[col_name] for col_name, kw in inputs.items()}
        elif isinstance(inputs, Sequence):
            # TODO: make this work with a list
            args = [data[col_name] for col_name in inputs]
            kwargs = {}
        elif inputs is None:
            # infer mapping from function signature if possible otherwise pass full row
            args = []
            kwargs = {}
            for name, param in signature(function).parameters.items():
                if name in data:
                    kwargs[name] = data[name]
                elif param.default is param.empty:
                    warnings.warn(
                        f"Non-default argument '{name}' does not have a corresponding "
                        "column in the DataFrame. If your function expects a full "
                        "DataFrame row, pass ``inputs='row'`` to ``map``. Otherwise, "
                        "please provide an `inputs` mapping "
                        "or pass a lambda function with a different signature. "
                        "See map documentation for more details.",
                    )
                    inputs = "row"
                    break

        if inputs == "row":
            args = []
            kwargs = {col_name: col for col_name, col in data.items()}

            def wrapper(*args, **kwargs):
                # FIXME: this should use data._clone instead!
                if is_batched_fn:
                    kwargs = DataFrame(kwargs)
                return base_function(kwargs)

            function = wrapper

        if args is None or kwargs is None:
            raise ValueError("``inputs`` must be Mapping, Sequence or 'row'")

    op = DeferredOp(
        fn=function,
        args=args,
        kwargs=kwargs,
        is_batched_fn=is_batched_fn,
        batch_size=batch_size,
        return_format=type(outputs) if outputs is not None else None,
        materialize_inputs=materialize,
    )

    block = DeferredBlock.from_block_data(data=op)

    first_row = op._get(0) if len(op) > 0 else None

    if outputs is None and isinstance(first_row, Dict):
        # support for splitting a dict into multiple columns without specifying outputs
        outputs = {output_key: output_key for output_key in first_row}
        op.return_format = type(outputs)

    if outputs is None and isinstance(first_row, Tuple):
        # support for splitting a tuple into multiple columns without specifying outputs
        outputs = tuple([str(i) for i in range(len(first_row))])
        op.return_format = type(outputs)

    if outputs is None or outputs == "single":
        # can only infer output type if the the input columns are nonempty
        if output_type is None and first_row is not None:
            output_type = infer_column_type([first_row])

        if not isinstance(output_type, Type):
            raise ValueError(
                "Must provide a single `output_type` if `outputs` is None."
            )

        col = DeferredColumn(
            data=BlockView(block_index=None, block=block), output_type=output_type
        )
        return col
    elif isinstance(outputs, Mapping):
        if output_type is None:
            output_type = {
                outputs[output_key]: infer_column_type([col])
                for output_key, col in first_row.items()
            }
        if not isinstance(output_type, Mapping):
            raise ValueError(
                "Must provide a `output_type` mapping if `outputs` is a mapping."
            )
        return DataFrame(
            {
                col: DeferredColumn(
                    data=BlockView(block_index=output_key, block=block),
                    output_type=output_type[outputs[output_key]],
                )
                for output_key, col in outputs.items()
            }
        )
    elif isinstance(outputs, Sequence):
        if output_type is None:
            output_type = [type(col) for col in first_row]
        if not isinstance(output_type, Sequence):
            raise ValueError(
                "Must provide a `output_type` sequence if `outputs` is a sequence."
            )
        return DataFrame(
            {
                col: DeferredColumn(
                    data=BlockView(block_index=output_key, block=block),
                    output_type=output_type[output_key],
                )
                for output_key, col in enumerate(outputs)
            }
        )


@docs.doc(source=_SHARED_DOCS_, data="data", name="defer")
def map(
    data: Union["DataFrame", "Column"],
    function: Callable,
    is_batched_fn: bool = False,
    batch_size: int = 1,
    inputs: Union[Mapping[str, str], Sequence[str]] = None,
    outputs: Union[Mapping[any, str], Sequence[str]] = None,
    output_type: Union[Mapping[str, Type["Column"]], Type["Column"]] = None,
    materialize: bool = True,
    use_ray: bool = False,
    num_blocks: int = 100,
    blocks_per_window: int = 10,
    pbar: bool = False,
    **kwargs,
):
    """Create a new :class:`Column` or :class:`DataFrame` by applying a
    function to each row in ${data}.

    This function shares nearly the exact same signature
    with :func:`defer`, the difference is that :func:`~meerkat.defer` returns a column
    that has not yet been computed. It is a placeholder for a column that will be
    computed later.

    Learn more in the user guide: :ref:`guide/dataframe/ops/mapping`.

    {input_description}

    *What gets returned by defer?*

    *   If ``function`` returns a single value, then ``defer``
        will return a :class:`DeferredColumn` object.

    *   If ``function`` returns a dictionary, then ``defer`` will return a
        :class:`DataFrame` containing :class:`DeferredColumn` objects. The keys of the
        dictionary are used as column names. The ``outputs`` argument can be used to
        override the column names.

    *   If ``function`` returns a tuple, then ``defer`` will return a :class:`DataFrame`
        containing :class:`DeferredColumn` objects. The column names will be integers.
        The column names can be overriden by passing a tuple to the ``outputs``
        argument.

    *   If ``function`` returns a tuple or a dictionary, then passing ``"single"`` to
        the ``outputs`` argument will cause ``defer`` to return a single
        :class:`DeferredColumn` that materializes to a :class:`ObjectColumn`

    .. note::
        This function is also available as a method of :class:`DataFrame` and
        :class:`Column` under the name ``map``.


    Args:
        ${data} (DataFrame): The :class:`DataFrame` or :class:`Column` to which the
            function will be applied.
        ${function}
        ${is_batched_fn}
        ${batch_size}
        ${inputs}
        ${outputs}
        ${output_type}
        ${materialize}
        use_ray (bool): Use Ray to parallelize the computation. Defaults to False.
        num_blocks (int): When using Ray, the number of blocks to split the data
            into. Defaults to 100.
        blocks_per_window (int): When using Ray, the number of blocks to process
            in a single Ray task. Defaults to 10.
        pbar (bool): Show a progress bar. Defaults to False.

    Returns:
        Union[DataFrame, Column]: A :class:`Column` or a :class:`DataFrame`.

    Examples
    ---------
    We start with a small DataFrame of voters with two columns: `birth_year`, which
    contains the birth year of each person, and `residence`, which contains the state in
    which each person lives.

    .. ipython:: python

        import datetime
        import meerkat as mk

        df = mk.DataFrame({
            "birth_year": [1967, 1993, 2010, 1985, 2007, 1990, 1943],
            "residence": ["MA", "LA", "NY", "NY", "MA", "MA", "LA"]
        })


    **Single input column.** Lazily create a column of birth years to a column of ages.

    .. ipython:: python

        df["age"] = df["birth_year"].map(
            lambda x: datetime.datetime.now().year - x
        )
        df["age"]


    **Multiple input columns.** Lazily create a column of birth years to a column of
    ages.

    .. ipython:: python

        df["ma_eligible"] = df.map(
            lambda age, residence: (residence == "MA") and (age >= 18)
        )
        df["ma_eligible"]
    """

    deferred = defer(
        data=data,
        function=function,
        is_batched_fn=is_batched_fn,
        batch_size=batch_size,
        inputs=inputs,
        outputs=outputs,
        output_type=output_type,
        materialize=materialize,
    )
    return _materialize(
        deferred,
        batch_size=batch_size,
        pbar=pbar,
        use_ray=use_ray,
        num_blocks=num_blocks,
        blocks_per_window=blocks_per_window,
    )


def _materialize(
    data: Union["DataFrame", "Column"],
    batch_size: int,
    pbar: bool,
    use_ray: bool,
    num_blocks: int,
    blocks_per_window: int,
):
    import logging

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import torch
    from tqdm import tqdm

    import meerkat as mk
    from meerkat.columns.abstract import column

    from .concat import concat

    if use_ray:
        import ray

        ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
        ray.data.set_progress_bars(enabled=pbar)

        # Step 1: Walk through the DeferredColumns and build a list of functions
        curr = data
        fns = []
        while isinstance(curr, mk.DeferredColumn):
            fns.append(curr.data.fn)

            # For linear pipelines, there will be either one elem in args or one key in
            # kwargs
            if curr.data.args:
                if len(curr.data.args) > 1:
                    raise ValueError("Multiple args not supported with `use_ray=True`.")
                curr = curr.data.args[0]
            elif curr.data.kwargs:
                if len(curr.data.kwargs) > 1:
                    raise ValueError(
                        "Multiple kwargs not supported with `use_ray=True`."
                    )
                curr = curr.data.kwargs[next(iter(curr.data.kwargs))]
            else:
                raise ValueError("No args or kwargs.")

        # Step 2: Create the ray dataset from the base column
        # TODO (dean): test added_dim on other data types
        added_dim = False
        if isinstance(curr, mk.PandasScalarColumn):
            ds = ray.data.from_pandas(pd.DataFrame({"0": curr})).repartition(num_blocks)
            fns.append(lambda x: x["0"])
        elif isinstance(curr, mk.ArrowScalarColumn):
            ds = ray.data.from_pandas(pa.table({"0": curr.data})).repartition(
                num_blocks
            )
            fns.append(lambda x: x["0"])
        elif isinstance(curr, mk.NumPyTensorColumn):
            ndarrays = curr.data
            if ndarrays.ndim == 1:
                added_dim = True
                ndarrays = np.expand_dims(ndarrays, 1)
            ds = ray.data.from_numpy(ndarrays).repartition(num_blocks)
        elif isinstance(curr, mk.TorchTensorColumn):
            ds = ray.data.from_torch(curr).repartition(num_blocks)
        elif isinstance(curr, mk.ObjectColumn):
            ds = ray.data.from_items(curr).repartition(num_blocks)
        elif isinstance(curr, mk.DataFrame):
            raise ValueError(
                "Multiple outputs (fan-out) not supported with `use_ray=True`."
            )
            # TODO (dean): Support fan-out (would have to create multiple pipelines)
            # ds = ray.data.from_pandas(curr.data._repr_pandas_()[0])
            # fns.append(lambda row: row.values())
        else:
            raise ValueError(
                f"Base column is of unsupported type {type(curr)} with `use_ray=True`."
            )

        # Step 3: Build the pipeline by walking backwards through fns
        pipe: ray.data.DatasetPipeline = ds.window(blocks_per_window=blocks_per_window)
        for fn in reversed(fns):
            # TODO (dean): if batch_size > 1, then use map_batches
            pipe = pipe.map(fn)

        # Step 4: Collect the results
        result_ds = iter(
            pipe.rewindow(blocks_per_window=num_blocks).iter_datasets()
        ).__next__()
        result = []
        if data._output_type == mk.NumPyTensorColumn:
            for partition in result_ds.to_numpy_refs():
                res = ray.get(partition)
                if len(res):
                    result.append(res[0][0] if added_dim else res[0])
            if added_dim:
                return mk.NumPyTensorColumn.from_array(result)
            return column(np.stack(result))
        elif data._output_type == mk.TorchTensorColumn:
            for partition in result_ds.to_torch():
                result.append(partition[0])
            return column(torch.stack(result))
        elif data._output_type == mk.PandasScalarColumn:
            for partition in result_ds.to_pandas_refs():
                result.append(ray.get(partition))
            return column(pd.concat(result)["value"])
        elif data._output_type == mk.ArrowScalarColumn:
            for partition in result_ds.to_arrow_refs():
                result.append(ray.get(partition)["value"].combine_chunks())
            return column(pa.concat_arrays(result))
        elif data._output_type == mk.ObjectColumn:
            for partition in result_ds.iter_batches():
                result.extend(partition)
            return column(result)
        else:
            raise ValueError(
                f"Unsupported output type {data._output_type} with `use_ray=True`."
            )

    else:

        result = []
        for batch_start in tqdm(range(0, len(data), batch_size), disable=not pbar):
            result.append(
                data._get(
                    slice(batch_start, batch_start + batch_size, 1), materialize=True
                )
            )
        return concat(result)
