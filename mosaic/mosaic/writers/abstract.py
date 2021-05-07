import abc


class AbstractWriter(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    @abc.abstractmethod
    def open(self, *args, **kwargs) -> None:
        return NotImplemented

    @abc.abstractmethod
    def write(self, data, *args, **kwargs) -> None:
        return NotImplemented

    @abc.abstractmethod
    def flush(self, *args, **kwargs) -> None:
        return NotImplemented

    @abc.abstractmethod
    def close(self, *args, **kwargs) -> None:
        return NotImplemented

    @abc.abstractmethod
    def finalize(self, *args, **kwargs) -> None:
        return NotImplemented
