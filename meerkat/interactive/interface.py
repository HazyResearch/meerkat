import code
import os
from functools import partial, wraps
from typing import Callable

from IPython.display import IFrame
from pydantic import BaseModel

from meerkat.constants import APP_DIR
from meerkat.interactive.app.src.lib.component.abstract import (
    Component,
    ComponentFrontend,
)
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state


def interface(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        interface = Interface(component=partial(fn, *args, **kwargs))
        return interface.launch()

    return wrapper


class InterfaceFrontend(BaseModel):
    component: ComponentFrontend
    name: str


class Interface(IdentifiableMixin):

    _self_identifiable_group: str = "interfaces"

    def __init__(
        self,
        component: Component,
        name: str = "Interface",
        id: str = None,
        height: str = "1000px",
        width: str = "100%",
    ):

        super().__init__(id=id)

        self.component = component
        self.name = name
        self.height = height
        self.width = width

        # self.write_component_wrappers()
        # self.write_sveltekit_route()

    def __call__(self):
        """Return the FastAPI object, this allows Interface objects
        to be targeted by uvicorn when running a script."""
        from meerkat.interactive.api import MeerkatAPI

        return MeerkatAPI

    def _to_svelte(self):

        # Check if we're in a Meerkat generated app
        # These apps are generated with the `mk init` command
        # They have a .mk file in the `app` directory
        import_prefix = "$lib"
        if os.path.exists(os.path.join(APP_DIR, ".mk")):
            # In a Meerkat generated app
            # Use the @meerkat-ml/meerkat package instead of $lib
            import_prefix = "@meerkat-ml/meerkat"

        all_components = list(sorted(self.component.get_components()))
        import_block = "\n".join(
            [
                f"    import {component} from '$lib/wrappers/{self.id}/{component}.svelte';"
                for component in all_components
            ]
        )

        component_mapping = [f"        {c}: {c}," for c in all_components]
        component_mapping = "\n".join(component_mapping)

        svelte = f"""\
<script lang="ts">
    import banner from '$lib/assets/banner_small.png';
    import {{ API_URL }} from '{import_prefix}';
    import {{ Interface }} from '{import_prefix}';
    import {{ onMount, setContext }} from 'svelte';

{import_block}

    setContext("Components", {{
{component_mapping}
    }})

    let config: Interface | null = null;
    onMount(async () => {{
        config = await (await fetch(`${{$API_URL}}/interface/{self.id}/config`)).json();
        document.title = "{self.name}";
    }});
</script>

<div class="h-screen p-3">
    {{#if config}}
        <Interface {{config}} />
    {{:else}}
        <div class="flex justify-center h-screen items-center">
            <img src={{banner}} alt="Meerkat" class="h-12" />
        </div>
    {{/if}}
</div>
"""
        return svelte

    def write_sveltekit_route(self):
        """Each Interface writes to a new SvelteKit route."""
        os.makedirs(f"{APP_DIR}/src/routes/{self.id}", exist_ok=True)
        # In dev mode, we allow reloading the same interface,
        # so we don't raise an error if the file already exists.
        # TODO: make this logic case on dev mode
        # if os.path.exists(f"{APP_DIR}/src/routes/{self.id}/+page.svelte"):
        #     raise ValueError(
        #         f"Interface with id {self.id} already exists. "
        #         "Please use a different id."
        #     )
        with open(f"{APP_DIR}/src/routes/{self.id}/+page.svelte", "w") as f:
            f.write(self._to_svelte())

        # TODO: Should rebuild the app here.
        # OR pass in --watch to vite build

    def _remove_svelte(self):
        # Remove all SvelteKit routes
        try:
            os.remove(f"{APP_DIR}/src/routes/{self.id}/+page.svelte")
        except OSError:
            pass
        try:
            os.rmdir(f"{APP_DIR}/src/routes/{self.id}")
        except OSError:
            pass

        # Remove all component wrappers
        for component_name in self.component.get_components():
            try:
                os.remove(
                    f"{APP_DIR}/src/lib/wrappers/{self.id}/{component_name}.svelte"
                )
            except OSError:
                pass

        try:
            os.rmdir(f"{APP_DIR}/src/lib/wrappers/{self.id}")
        except OSError:
            pass

    def launch(self, return_url: bool = False):
        from meerkat.interactive.startup import is_notebook, output_startup_message

        if state.frontend_info is None:
            raise ValueError(
                "Interactive mode not initialized. "
                "Run `network = mk.gui.start()` first."
            )

        url = f"{state.frontend_info.url}/{self.id}"

        if return_url:
            return url
        if is_notebook():
            return IFrame(url, width=self.width, height=self.height)
        else:
            import webbrowser

            output_startup_message(url=url, docs_url=state.api_info.docs_url)
            webbrowser.open(url)

            # get locals of the main module when running in script.
            import __main__

            code.interact(local=__main__.__dict__)

    @property
    def frontend(self):
        return InterfaceFrontend(name=self.name, component=self.component.frontend)
