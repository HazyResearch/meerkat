class MergeError(ValueError):
    pass


class ConcatError(ValueError):
    pass


class ConcatWarning(RuntimeWarning):
    pass


class ConsolidationError(ValueError):
    pass


class ImmutableError(ValueError):
    pass


class ConversionError(ValueError):
    pass


class ExperimentalWarning(FutureWarning):
    pass


class TriggerError(Exception):
    pass
