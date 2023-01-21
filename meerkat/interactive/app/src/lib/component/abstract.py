import collections
import inspect
import os
from typing import Dict, List, Literal, Set

from pydantic import BaseModel, Extra, root_validator, validator

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, EndpointProperty
from meerkat.interactive.frontend import FrontendMixin
from meerkat.interactive.graph import Store
from meerkat.interactive.node import Node, NodeMixin
from meerkat.mixins.identifiable import IdentifiableMixin, classproperty
from meerkat.tools.utils import nested_apply


class ComponentFrontend(BaseModel):
    component_id: str
    path: str
    name: str
    props: Dict
    slots: list
    library: str


class WrappableMixin:
    @classproperty
    def wrapper_import_style(cls) -> Literal["default", "named", "none"]:
        from meerkat.interactive.svelte import SvelteWriter

        svelte_writer = SvelteWriter()

        if cls.library == "@meerkat-ml/meerkat" and cls.namespace == "meerkat":
            # Meerkat components
            if not svelte_writer.is_user_appdir:
                # In Meerkat package
                # Use named import: import Something from "path/to/component";
                return "named"
            else:
                # Use default import: import { Something } from "@meerkat-ml/meerkat";
                return "default"
        elif cls.library == "@meerkat-ml/meerkat":
            # Custom user components
            if svelte_writer.is_user_appdir:
                # Use named import: import Something from "path/to/component";
                return "named"
            else:
                # This should never happen: we should never be wrapping a custom
                # component directly in the Meerkat package + no components
                # in Meerkat should have a namespace other than "meerkat"
                raise ValueError(
                    f"Cannot use custom component {cls.component_name}, "
                    "please initialize a Meerkat app using `mk init` first."
                )
        elif cls.library == "html":
            # No need to import HTML tags
            return "none"
        else:
            return "default"


class PythonToSvelteMixin:
    def get_components(self) -> Set[str]:
        nested_components = set()
        nested_components.add(self.component_name)
        for s in self.slots:
            nested_components.update(s.get_components())
        if hasattr(self, "component"):
            nested_components.update(self.component.get_components())
        if hasattr(self, "components"):
            for c in self.components:
                nested_components.update(c.get_components())
        return nested_components


class Slottable:
    @classproperty
    def slottable(cls) -> bool:
        return True


def iterable(arg):
    return isinstance(arg, collections.Iterable) and not isinstance(arg, str)


class SlotsMixin:
    def __init__(self, slots: List["Component"] = [], *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not iterable(slots):
            slots = [slots]

        self._slots = slots

    @property
    def slots(self) -> List["Component"]:
        from meerkat.interactive.app.src.lib.layouts import Brace

        _slots = []
        for slot in self._slots:
            if not isinstance(slot, Component):
                # Wrap it in a Brace component
                _slots.append(Brace(data=slot))
            else:
                _slots.append(slot)
        return _slots

    @classproperty
    def slottable(cls) -> bool:
        return False


class Component(
    IdentifiableMixin,
    FrontendMixin,
    SlotsMixin,
    WrappableMixin,
    PythonToSvelteMixin,
    BaseModel,
):
    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, Node):
            # because the validator converts dataframes to nodes, when the
            # dataframe is accessed we need to convert it back to the dataframe
            return value.obj
        return value

    @classproperty
    def alias(cls):
        """Unique alias for this component that uses the namespace and the name
        of the Component subclass.

        This will give components with the same name from different
        libraries different names e.g. `MeerkatButton` and
        `CarbonButton`.
        """
        return cls.namespace.title() + cls.__name__

    @classproperty
    def component_name(cls):
        # Inheriting an existing Component and modifying it on the Python side
        # should not change the name of the component used on the frontend
        if cls.__bases__[0] != Component and issubclass(cls.__bases__[0], Component):
            return cls.__bases__[0].__name__

        return cls.__name__

    @classproperty
    def event_names(cls) -> List[str]:
        """Returns a list of event names that this component emits."""
        return [
            k[3:]
            for k in cls.__fields__
            if k.startswith("on_")
            and not issubclass(cls.__fields__[k].type_, EndpointProperty)
        ]

    @classproperty
    def events(cls) -> List[str]:
        """Returns a list of events that this component emits."""
        return [
            k
            for k in cls.__fields__
            if k.startswith("on_")
            and not issubclass(cls.__fields__[k].type_, EndpointProperty)
        ]

    @classproperty
    def frontend_alias(cls):
        """Alias for this component that is used in the frontend.

        This is not unique, and it is possible to have multiple
        components with the same frontend alias. This is useful for
        components that are just wrappers around other components, e.g.
        a layout Component that subclasses a Grid Component will still
        have the same frontend alias as the Grid Component.
        """
        return cls.namespace.title() + cls.component_name

    @classproperty
    def identifiable_group(self):
        # Ordinarily, we would create a new classproperty for this, like
        # _self_identifiable_group: str = "components"
        # However, this causes pydantic to show _self_identifiable_group in
        # type hints when using the component in the IDE, which might
        # be confusing to users.
        # We just override the classproperty here directly as an alternative.
        return "components"

    @classproperty
    def library(cls):
        return "@meerkat-ml/meerkat"

    @classproperty
    def namespace(cls):
        return "meerkat"

    @classproperty
    def path(cls):
        from meerkat.interactive.svelte import svelte_writer

        if not cls.library == "@meerkat-ml/meerkat" or (
            cls.library == "@meerkat-ml/meerkat"
            and cls.namespace == "meerkat"
            and svelte_writer.is_user_appdir
        ):
            return cls.library

        path = os.path.join(
            os.path.dirname(inspect.getfile(cls)),
            f"{cls.component_name}.svelte",
        )
        if os.path.exists(path):
            return path

        # Raise an error if the file doesn't exist
        raise FileNotFoundError(
            f"Could not find {path}. "
            f"Check that the definition of this Component {cls} "
            "is in the same folder as the Svelte file. "
            "You might also be using a "
            "component from a library, in which case set the `library` "
            "property of the Component correctly."
        )

    @classproperty
    def prop_names(cls):
        return [
            k for k in cls.__fields__ if not k.startswith("on_") and "_self_id" != k
        ] + [
            k
            for k in cls.__fields__
            if k.startswith("on_")
            and issubclass(cls.__fields__[k].type_, EndpointProperty)
        ]

    @classproperty
    def prop_bindings(cls):
        if not issubclass(cls, AutoComponent):
            # These props need to be bound with `bind:` in Svelte
            types_to_bind = {Store, DataFrame}
            return {
                prop: cls.__fields__[prop].type_ in types_to_bind
                for prop in cls.prop_names
            }
        else:
            return {
                prop: (cls.__fields__[prop].type_ != EndpointProperty)
                for prop in cls.prop_names
            }

    @property
    def frontend(self):
        def _frontend(value):
            if isinstance(value, FrontendMixin):
                return value.frontend
            return value

        frontend_props = nested_apply(
            self.virtual_props,
            _frontend,
            base_types=(Store),
        )

        return ComponentFrontend(
            component_id=self.id,
            path=os.path.join(
                os.path.dirname(inspect.getfile(self.__class__)),
                f"{self.component_name}.svelte",
            ),
            name=self.alias,
            props=frontend_props,
            slots=[slot.frontend for slot in self.slots],
            library=self.library,
        )

    @property
    def props(self):
        return {k: self.__getattribute__(k) for k in self.prop_names}

    @property
    def virtual_props(self):
        """Props, and all events (as_*) as props."""
        vprop_names = [k for k in self.__fields__ if "_self_id" != k]
        return {k: self.__getattribute__(k) for k in vprop_names}

    @validator("*", pre=False)
    def _check_inode(cls, value):
        if isinstance(value, NodeMixin) and not isinstance(value, Store):
            # Now value is a NodeMixin object
            # We need to make sure that value points to a Node in the graph
            # If it doesn't, we need to add it to the graph
            if not value.has_inode():
                value.attach_to_inode(value.create_inode())

            # Now value is a NodeMixin object that points to a Node in the graph

            # We replace `value` with `value.inode`, and will send
            # this to the frontend
            # Effectively, NodeMixin objects (except Store) are "by reference"
            # and not "by value" (this is also why we explicitly exclude
            # Store from this check, which is "by value")
            return value.inode
        return value

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow
        copy_on_model_validation = False


class AutoComponent(Component):
    """Component with simple defaults."""

    @classproperty
    def component_name(cls):
        # Inheriting an existing AutoComponent and modifying it on the Python side
        # should not change the name of the component used on the frontend
        if cls.__bases__[0] != AutoComponent and issubclass(
            cls.__bases__[0], AutoComponent
        ):
            return cls.__bases__[0].__name__

        return cls.__name__

    @root_validator(pre=False)
    def _convert_fields(cls, values):
        for name, value in values.items():
            # Wrap all the fields that are not NodeMixins in a Store
            # (i.e. this will exclude DataFrame, Endpoint etc. as well as
            # fields that are already Stores)
            if (
                cls.__fields__[name].type_ == Endpoint
                or cls.__fields__[name].type_ == EndpointProperty
            ):
                # Separately skip Endpoint fields by looking at the field type,
                # since they are assigned None by default and would be missed
                # by the condition below
                continue

            if not isinstance(value, NodeMixin) and not isinstance(value, Node):
                value = values[name] = Store(value)

            # Now make sure that all the `Store` objects have inodes
            if hasattr(value, "has_inode") and not value.has_inode():
                value.attach_to_inode(value.create_inode())

        return values
