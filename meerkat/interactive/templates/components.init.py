from meerkat import classproperty
from meerkat.interactive import Component


class LibraryMixin:
    @classproperty
    def namespace(cls):
        return "custom"


# Component should always be the last base class
class ExampleComponent(LibraryMixin, Component):
    """Make custom components by extending the `Component` class.

    Add the `namespace` and `library` class properties, so that
    it's easy to import and publish your components. Read our
    guide on how to publish your components to learn more.

    The component definition should reflect the structure of the
    frontend component, and its props.

    Make sure that
    - all custom components are defined inside this `components` directory,
    or any subdirectories.
    - any `.svelte` Component file defines a corresponding Python class
    in a `.py` file in the same directory.
    - all components you define in any subdirectories are imported
    in this `__init__.py` file of the `components` directory. Meerkat
    automatically imports and registers all components defined in this
    file.

    For fine-grained control of which fields are synchronized
    with the frontend, you can directly extend the `Component` class instead
    of the `Component` class. See our documentation for details.
    """

    name: str = "World"
