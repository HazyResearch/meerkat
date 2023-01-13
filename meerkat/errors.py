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

class TriggerError(ValueError):
   # pass
    def __init__(self, status, msg):
        self.status = status
        self.msg = msg
        super().__init__(msg);
   

