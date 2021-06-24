from typing import Any, Dict


class CloneableMixin:
    def _clone_kwargs(self) -> Dict[str, Any]:
        """Returns __init__ kwargs for instantiating new object.

        This function returns the default parameters that should be plumbed
        from the current instance to the new instance.

        This is the API that should be used by DataPanel and AbstractColumn
        subclasses that require unique protocols for instantiation.

        Returns:
            Dict[str, Any]: The keyword arguments for initialization.
                These arguments will be used by :meth:`_clone`.
        """
        raise NotImplementedError()

    def _clone(self, data=None, **kwargs):
        default_kwargs = self._clone_kwargs()
        if data is None:
            data = kwargs.pop("data", self.data)
        if kwargs:
            default_kwargs.update(kwargs)
        return self.__class__(data, **default_kwargs)
