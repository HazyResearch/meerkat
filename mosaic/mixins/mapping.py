import logging
from typing import Callable, Mapping, Optional

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class MappableMixin:
    def __init__(self, *args, **kwargs):
        super(MappableMixin, self).__init__(*args, **kwargs)

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_workers: Optional[int] = 0,
        output_type: type = None,
        mmap: bool = False,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ):
        # TODO (sabri): add materialize?
        from mosaic.columns.abstract import AbstractColumn
        from mosaic.datapanel import DataPanel

        """Map a function over the elements of the column."""
        # Check if need to materialize:
        # TODO(karan): figure out if we need materialize=False

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Ensure that num_workers is not None
        if num_workers is None:
            num_workers = 0

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        if not batched:
            # Convert to a batch function
            function = self._convert_to_batch_fn(
                function, with_indices=with_indices, materialize=materialize
            )
            batched = True
            logger.info(f"Converting `function` {function} to a batched function.")

        # Run the map
        logger.info("Running `map`, the dataset will be left unchanged.")
        for i, batch in tqdm(
            enumerate(
                self.batch(
                    batch_size=batch_size,
                    drop_last_batch=drop_last_batch,
                    num_workers=num_workers,
                    materialize=materialize
                    # TODO: collate=batched was commented out in list_column
                )
            ),
            total=(len(self) // batch_size)
            + int(not drop_last_batch and len(self) % batch_size != 0),
            disable=not pbar,
        ):
            # Calculate the start and end indexes for the batch
            start_index = i * batch_size
            end_index = min(len(self), (i + 1) * batch_size)

            # Use the first batch for setup
            if i == 0:
                # Get some information about the function
                function_properties = self._inspect_function(
                    function,
                    with_indices,
                    batched,
                    batch,
                    range(start_index, end_index),
                    materialize=materialize,
                )

                # Pull out information
                output = function_properties.output
                dtype = function_properties.output_dtype
                is_mapping = isinstance(output, Mapping)
                writers, output_types = {}, {}
                for key, curr_output in output.items() if is_mapping else [(0, output)]:
                    curr_output_type = (
                        type(AbstractColumn.from_data(curr_output))
                        if output_type is None
                        else output_type
                    )
                    output_types[key] = curr_output_type
                    writer = curr_output_type.get_writer(mmap=mmap)

                    # Setup for writing to a certain output column
                    # TODO: support optionally memmapping only some columns
                    if mmap:
                        shape = (len(self), *curr_output.shape)
                        # Construct the mmap file path
                        # TODO: how/where to store the files
                        path = self.logdir / f"{hash(function)}" / key
                        # Open the output writer
                        writer.open(str(path), dtype, shape=shape)
                    else:
                        # Create an empty dict or list for the outputs
                        writer.open()
                    writers[key] = writer

            else:
                # Run `function` on the batch
                output = (
                    function(
                        batch,
                        range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                    )
                    if with_indices
                    else function(batch)
                )

            # Append the output
            if output is not None:
                if isinstance(output, Mapping):
                    if set(output.keys()) != set(writers.keys()):
                        raise ValueError(
                            "Map function must return same keys for each batch."
                        )
                    for k, writer in writers.items():
                        writer.write(output[k])
                else:
                    writers[0].write(output)

        # Check if we are returning a special output type
        outputs = {}
        for key, writer in writers.items():
            if mmap:
                writer.flush()
                # TODO: Writers should have correspondence to their own columns
                outputs[key] = output_types[key].read(
                    str(path),
                    mmap=mmap,
                    dtype=dtype,
                    shape=shape,
                )
            else:
                data = writer.flush()
                outputs[key] = output_types[key].from_data(data)

        if not is_mapping:
            outputs = outputs[0]
        else:
            outputs = DataPanel.from_batch(outputs)
        return outputs
