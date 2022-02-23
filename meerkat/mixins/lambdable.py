import logging
from typing import Callable, Dict, Optional, Tuple, Union
from meerkat.block.abstract import BlockView


logger = logging.getLogger(__name__)


class LambdaMixin:
    def __init__(self, *args, **kwargs):
        super(LambdaMixin, self).__init__(*args, **kwargs)

    def to_lambda(
        self,
        function: Callable,
        is_batched_fn: bool = False,
        batch_size: int = 1,
        outputs: Union[Dict[any, str], Tuple[str]] = None,
        output_type: Union[Dict[str, type], type] = None
    ):
        from meerkat.columns.abstract import AbstractColumn
        from meerkat.datapanel import DataPanel
        from meerkat import LambdaColumn
        from meerkat.block.lambda_block import LambdaOp, LambdaBlock

        if isinstance(self, AbstractColumn):
            args = [self]
            kwargs = {}
        elif isinstance(self, DataPanel):
            inputs = {}

        op = LambdaOp(
            fn=function,
            args=args,
            kwargs=kwargs,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            return_format=type(outputs)
        )
        # TODO: make lambda op

        block = LambdaBlock.from_block_data(data=op)

        if (outputs is None):
            if not isinstance(output_type, type):
                raise ValueError

            col = LambdaColumn(
                data=BlockView(block_index=None, block=block), output_type=output_type
            )
            return col
        elif isinstance(outputs, Dict):
            return DataPanel(
                {
                    col: LambdaColumn(
                        data=BlockView(block_index=output_key, block=block),
                        output_type=output_type[output_key]
                    )
                    for output_key, col in outputs.items()
                }
            )
        elif isinstance(outputs, Tuple):
            return DataPanel(
                {
                    col: LambdaColumn(data=BlockView(block_index=output_key, block=block))
                    for output_key, col in enumerate(outputs)
                }
            )
