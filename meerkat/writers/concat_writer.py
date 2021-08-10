from meerkat.columns.abstract import AbstractColumn
from meerkat.writers.abstract import AbstractWriter


class ConcatWriter(AbstractWriter):
    def __init__(
        self,
        output_type: type = AbstractColumn,
        template: AbstractColumn = False,
        *args,
        **kwargs,
    ):
        super(ConcatWriter, self).__init__(*args, **kwargs)
        self.output_type = output_type
        self.template = template

    def open(self) -> None:
        self.outputs = []

    def write(self, data, **kwargs) -> None:
        # convert to Meerkat column if not already
        if self.template is not None:
            if isinstance(data, AbstractColumn):
                data = data.data
            data = self.template._clone(data=data)
        elif not isinstance(data, AbstractColumn):
            data = self.output_type(data)

        self.outputs.append(data)

    def flush(self, *args, **kwargs):
        from meerkat.ops.concat import concat

        return concat(self.outputs)

    def close(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs) -> None:
        pass
