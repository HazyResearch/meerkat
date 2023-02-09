class Singleton(type):
    """A metaclass that ensures only one instance of a class is created.

    Usage:

        >>> class Example(metaclass=Singleton):
        ...     def __init__(self, x):
        ...         self.x = x
        >>> a = Example(1)
        >>> b = Example(2)
        >>> print(a, id(a))
        <__main__.Example object at 0x7f8b8c0b7a90> 140071000000000
        >>> print(b, id(b))
        <__main__.Example object at 0x7f8b8c0b7a90> 140071000000000
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        # Look up if this cls pair has been created before
        if cls not in cls._instances:
            # If not, we let a new instance be created
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
