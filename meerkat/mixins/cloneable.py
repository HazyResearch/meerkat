from typing import Any, Dict


class CloneableMixin:
    def _clone_kwargs(self) -> Dict[str, Any]:
        """Returns __init__ kwargs for instantiating new object.

        This function returns the default parameters that should be plumbed
        from the current instance to the new instance.

        This is the API that should be used by DataPanel and AbstractColumn
        subclasses that require unique protocols for instantiation.

        When subclassing you can update the _clone_kwargs with additional kwargs. e.g. 
        ```
        class EntityDataPanel:
            def _clone_kwargs(self) -> EntityDataPanel:
                default_kwargs = super()._clone_kwargs()
                default_kwargs.update({"index_column": self.index_column})
                return default_kwargs
        ```

        Returns:
            Dict[str, Any]: The keyword arguments for initialization.
                These arguments will be used by :meth:`_clone`.
        """
        raise NotImplementedError()

    def _clone(self, data=None, **kwargs):
        default_kwargs = self._clone_kwargs()

        if kwargs:
            default_kwargs.update(kwargs)
        
        if data is None:
            data = default_kwargs.pop("data", self.data)
        else:
            default_kwargs.pop("data", None)

        return self.__class__(data, **default_kwargs)
