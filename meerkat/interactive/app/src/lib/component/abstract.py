import inspect
import os
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Extra, validator, root_validator
from pydantic.fields import ModelField
from meerkat.constants import APP_DIR
from meerkat.dataframe import DataFrame

from meerkat.interactive.endpoint import Endpoint
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
    @property
    def wrapped_path(self):
        return self.path

    @property
    def wrapper_import_style(self) -> Literal["default", "named"]:
        if self.library == "@meerkat-ml/meerkat":
            return "named"
        else:
            return "default"

    def _to_svelte_wrapper(self):
        if self.wrapper_import_style == "named":
            import_statement = f"import {self.component_name} from '{self.path}';"
        else:
            import_statement = f"import {{ {self.component_name} }} from '{self.path}';"

        prop_exports = "\n".join([f"    export let {prop};" for prop in self.props])
        event_exports = "\n".join(
            [f"    export let on_{event};" for event in self.events()]
        )

        prop_bindings = " ".join(
            [
                "bind:" + prop + "={$" + prop + "}"
                if self.__fields__[prop].type_ == Store
                or self.__fields__[prop].type_ == DataFrame
                else "{" + prop + "}"
                for prop in self.props
            ]
        )

        event_bindings = " ".join(
            [
                f"on:{event}={{ (e) => on_{event} ? dispatch(on_{event}.endpoint_id, {{detail: e.detail}}) : null }}"
                for event in self.events()
            ]
        )

        return f"""\
<script>
    {import_statement}
    import {{ getContext }} from 'svelte';
	const {{ dispatch }} = getContext('Meerkat');
        
{prop_exports}

{event_exports}
</script>

<{self.component_name} {prop_bindings} {event_bindings}>
    {"<slot />" if self.slottable else ""}
</{self.component_name}>
"""

    def to_svelte_wrapper(self):
        wrappers = {}
        wrappers[self.component_name] = self._to_svelte_wrapper()
        for s in self.slots:
            wrappers.update(s.to_svelte_wrapper())
        if hasattr(self, "component"):
            wrappers.update(self.component.to_svelte_wrapper())
        if hasattr(self, "components"):
            for c in self.components:
                wrappers.update(c.to_svelte_wrapper())
        return wrappers


class PythonToSvelteMixin:
    def to_svelte_imports(self):
        nested_imports = []
        for s in self.slots:
            nested_imports.extend(s.to_svelte_imports())
        if hasattr(self, "component"):
            nested_imports.extend(self.component.to_svelte_imports())
        if hasattr(self, "components"):
            for c in self.components:
                nested_imports.extend(c.to_svelte_imports())
        return [
            f'    import {self.component_name} from "{self.wrapped_path}";',
        ] + nested_imports

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

    def to_svelte_import_block(self):
        imports = sorted(list(set(self.to_svelte_imports())))
        return "\n".join(imports)

    def to_svelte_script(self):
        return f"""\
<script lang="ts">
{self.to_svelte_import_block()}
</script>\
"""

    def to_svelte_markup(self, indent=""):
        return f"""\
{indent}<{self.component_name} \
{" ".join(["bind:" + "{$" +  prop + "}"
if self.__fields__[prop].type_ == Store 
else "{" + prop + "}" for prop in self.props])}>
{"".join([s.to_svelte_markup(indent=indent + "    ") for s in self.slots])}\
{indent}</{self.component_name}>
"""

    def to_svelte(self):
        return f"""\
{self.to_svelte_script()}

{self.to_svelte_markup()}\
"""


class Slottable:
    @property
    def slottable(self) -> bool:
        return True


class SlotsMixin:
    def __init__(self, slots: List["Component"] = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slots = slots

    @property
    def slots(self) -> List["Component"]:
        return self._slots

    @property
    def slottable(self) -> bool:
        return False


class EventsMixin:
    @classmethod
    def events(cls) -> List[str]:
        return []


class Component(
    IdentifiableMixin,
    FrontendMixin,
    SlotsMixin,
    EventsMixin,
    WrappableMixin,
    PythonToSvelteMixin,
    BaseModel,
):
    @classmethod
    def add_fields(cls, **field_definitions: Any):
        new_fields: Dict[str, ModelField] = {}
        new_annotations: Dict[str, Optional[type]] = {}
        
        for f_name, f_def in field_definitions.items():
            if isinstance(f_def, tuple):
                try:
                    f_annotation, f_value = f_def
                except ValueError as e:
                    raise Exception(
                        "field definitions should either be a tuple of (<type>, <default>) or just a "
                        "default value, unfortunately this means tuples as "
                        "default values are not allowed"
                    ) from e
            else:
                f_annotation, f_value = None, f_def

            if f_annotation:
                new_annotations[f_name] = f_annotation
            
            new_fields[f_name] = ModelField.infer(
                name=f_name,
                value=f_value,
                annotation=f_annotation,
                class_validators=None,
                config=cls.__config__,
            )

        cls.__fields__.update(new_fields)
        cls.__annotations__.update(new_annotations)

    @property
    def library(self):
        return "@meerkat-ml/meerkat"

    @property
    def path(self):
        if not self.library == "@meerkat-ml/meerkat":
            return self.library

        path = os.path.join(
            os.path.dirname(inspect.getfile(self.__class__)),
            f"{self.component_name}.svelte",
        )
        if os.path.exists(path):
            return path

        # Raise an error if the file doesn't exist
        raise FileNotFoundError(
            f"Could not find {path}. "
            f"Check that the definition of this Component {self} "
            "is in the same folder as the Svelte file. "
            "You might also be using a "
            "component from a library, in which case set the `library` "
            "property of the Component correctly."
        )

    @property
    def component_name(self):
        # Inheriting an existing Component and modifying it on the Python side
        # should not change the name of the component used on the frontend
        if self.__class__.__bases__[0] != Component and issubclass(
            self.__class__.__bases__[0], Component
        ):
            return self.__class__.__bases__[0].__name__

        return self.__class__.__name__

    @classproperty
    def identifiable_group(self):
        # Ordinarily, we would create a new classproperty for this, like
        # _self_identifiable_group: str = "components"
        # However, this causes pydantic to show _self_identifiable_group in
        # type hints when using the component in the IDE, which might
        # be confusing to users.
        # We just override the classproperty here directly as an alternative.
        return "components"

    @root_validator(pre=True)
    def _dummy_init(cls, values):
        events = cls.events()
        if events:
            cls.add_fields(**{f"on_{event}": (Endpoint, None) for event in events})
        return values

    @validator("*", pre=False)
    def _check_inode(cls, value):
        if isinstance(value, NodeMixin) and not isinstance(value, Store):
            # Now value is a NodeMixin object
            # We need to make sure that value points to a Node in the graph
            # If it doesn't, we need to add it to the graph
            if not value.has_inode():
                value.attach_to_inode(value.create_inode())

            # Now value is a NodeMixin object that points to a Node in the graph
            return value.inode  # this will exist
        return value

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, Node):
            # because the validator above converts dataframes to nodes, when the
            # dataframe is accessed we need to convert it back to the dataframe
            return value.obj
        return value

    @property
    def props(self):
        prop_names = [k for k in self.__fields__ if not k.startswith("on_") and "_self_id" != k]
        return {k: self.__getattribute__(k) for k in prop_names}
    
    @property
    def virtual_props(self):
        """Props, and all events (as_*) as props."""
        prop_names = [k for k in self.__fields__ if "_self_id" != k]
        return {k: self.__getattribute__(k) for k in prop_names}

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
            name=self.component_name,
            props=frontend_props,
            slots=[slot.frontend for slot in self.slots],
            library=self.library,
        )

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow
        copy_on_model_validation = False
