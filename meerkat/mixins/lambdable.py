import logging
from multiprocessing.sharedctypes import Value
from typing import Callable, Mapping, Sequence, Union, TYPE_CHECKING

from meerkat.block.abstract import BlockView

if TYPE_CHECKING:
    from meerkat.columns.lambda_column import LambdaColumn
    from meerkat.datapanel import DataPanel

logger = logging.getLogger(__name__)


class LambdaMixin:
    def __init__(self, *args, **kwargs):
        super(LambdaMixin, self).__init__(*args, **kwargs)

    def to_lambda(
        self,
        function: Callable,
        is_batched_fn: bool = False,
        batch_size: int = 1,
        inputs: Union[Mapping[str, str], Sequence[str]] = None,
        outputs: Union[Mapping[any, str], Sequence[str]] = None,
        output_type: Union[Mapping[str, type], type] = None,
    ) -> Union["DataPanel", "LambdaColumn"]:
        """_summary_

        Examples
        ---------


        Args:
            function (Callable): _description_
            is_batched_fn (bool, optional): _description_. Defaults to False.
            batch_size (int, optional): Minimum batch_size . Defaults to 1.
            inputs (Dict[str, str], optional): Dictionary mapping column names in
                ``self`` to keyword arguments of ``function``. Ignored if ``self`` is a
                column. When calling ``function`` values from the columns will be fed to
                the corresponding keyword arguments. Defaults to None, in which case the
                entire datapanel.
            outputs (Union[Dict[any, str], Tuple[str]], optional): _description_. Defaults to None.
            output_type (Union[Dict[str, type], type], optional): _description_. Defaults to None.

        Examples

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        from meerkat.columns.abstract import AbstractColumn
        from meerkat.datapanel import DataPanel
        from meerkat import LambdaColumn
        from meerkat.block.lambda_block import LambdaOp, LambdaBlock

        # prepare arguments for LambdaOp
        if isinstance(self, AbstractColumn):
            args = [self]
            kwargs = {}
        elif isinstance(self, DataPanel):
            if isinstance(inputs, Mapping):
                args = []
                kwargs = {kw: self[col_name] for col_name, kw in inputs.items()}
            elif isinstance(inputs, Sequence):
                args = [self[col_name] for col_name in inputs]
                kwargs = {}
            elif inputs is None:
                args = [self]
                kwargs = {}
            else:
                raise ValueError("")

        op = LambdaOp(
            fn=function,
            args=args,
            kwargs=kwargs,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            return_format=type(outputs),
        )

        block = LambdaBlock.from_block_data(data=op)

        if outputs is None:
            if not (isinstance(output_type, type) or output_type is None):
                raise ValueError

            col = LambdaColumn(
                data=BlockView(block_index=None, block=block), output_type=output_type
            )
            return col
        elif isinstance(outputs, Mapping):
            return DataPanel(
                {
                    col: LambdaColumn(
                        data=BlockView(block_index=output_key, block=block),
                        output_type=output_type[output_key],
                    )
                    for output_key, col in outputs.items()
                }
            )
        elif isinstance(outputs, Sequence):
            return DataPanel(
                {
                    col: LambdaColumn(
                        data=BlockView(block_index=output_key, block=block)
                    )
                    for output_key, col in enumerate(outputs)
                }
            )
