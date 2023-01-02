import dataclasses
import importlib
import os
import subprocess
import json
import shutil
import sys
from typing import TYPE_CHECKING, List, Literal, Set, Type

from jinja2 import Environment, FileSystemLoader

from meerkat.constants import APP_DIR, BASE_DIR
from meerkat.interactive import Component
from meerkat.interactive.app.src.lib.component.abstract import AutoComponent


if TYPE_CHECKING:
    from meerkat.interactive import Interface

NPM_PACKAGE = "@meerkat-ml/meerkat"

jinja_env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates"))
)


@dataclasses.dataclass
class SvelteWriter:
    """
    Class that handles writing Svelte components to the Meerkat app.
    """

    appname: str = "meerkat_app"
    cwd: str = dataclasses.field(default_factory=os.getcwd)
    package_manager: Literal["bun", "npm"] = "npm"
    _appdir: str = None

    def __post_init__(self):
        if self._appdir:
            # User defined appdir
            return
        if os.path.exists(os.path.join(self.cwd, "app")):
            # Otherwise, check if we're in a Meerkat generated app
            self._appdir = os.path.join(self.cwd, "app")
        else:
            self._appdir = APP_DIR

    @property
    def appdir(self):
        return self._appdir

    @property
    def is_user_appdir(self) -> bool:
        """Check if a Meerkat generated app can be used.
        These apps are generated with the `mk init` command."""
        if os.path.exists(os.path.join(self.appdir, ".mk")):
            return True
        return False

    def add_svelte(self):
        return subprocess.run(
            [self.package_manager, "add", "create-svelte@latest"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def copy_banner_small(self):
        """Copy the Meerkat banner to the new app."""
        # Get path to banner at
        # "meerkat/interactive/app/src/lib/assets/banner_small.png"
        banner_path = os.path.join(
            BASE_DIR,
            "meerkat",
            "interactive",
            "app",
            "src",
            "lib",
            "assets",
            "banner_small.png",
        )

        # Copy banner to the new app
        dir = f"{self.appdir}/src/lib/assets"
        os.makedirs(dir, exist_ok=True)
        shutil.copy(banner_path, f"{dir}/banner_small.png")

    def copy_favicon(self):
        """Copy the Meerkat favicon to the new app."""
        # Get path to favicon.png, at "meerkat/interactive/app/static/favicon.png"
        favicon_path = os.path.join(
            BASE_DIR,
            "meerkat",
            "interactive",
            "app",
            "static",
            "favicon.png",
        )

        # Copy favicon.png to the new app
        shutil.copy(favicon_path, f"{self.appdir}/static/favicon.png")

    def delete_installer(self):
        """Delete the Meerkat app installer directory."""
        return subprocess.run(["rm", "-rf", "installer"])

    def get_all_components(
        self,
        exclude_classes: Set[str] = {"AutoComponent", "Component"},
    ) -> List[Type["Component"]]:
        """Get all subclasses of Component, excluding the ones in
        `exclude_classes`.

        Args:
            exclude_classes (Set[str], optional): Set of classes
                to exclude. Defaults to {"AutoComponent", "Component"}.

        Returns:
            List[Type["Component"]]: List of subclasses of Component.
        """
        from meerkat.interactive.startup import get_subclasses_recursive

        # Import user components
        self.import_app_components()

        # Recursively find all subclasses of Component
        subclasses = get_subclasses_recursive(Component)

        # Filter out the classes we don't want and sort
        subclasses = [c for c in subclasses if c.__name__ not in exclude_classes]
        subclasses = sorted(subclasses, key=lambda c: c.alias)

        return subclasses

    def get_all_frontend_components(self) -> List[Type["Component"]]:
        # Create a `frontend_components` list that contains the
        # components that have unique component.frontend_alias
        components = self.get_all_components()
        frontend_components = []
        aliases = set()
        for component in components:
            if component.frontend_alias not in aliases:
                frontend_components.append(component)
                aliases.add(component.frontend_alias)

        return frontend_components

    def import_app_components(self):
        if self.is_user_appdir:
            # Import all components inside the app/src/lib/components
            # directory to register user components from the app
            # Otherwise do nothing
            sys.path.append(self.cwd)
            importlib.import_module("app.src.lib.components")

    def install_bun(self):
        return subprocess.run(
            ["curl https://bun.sh/install | sh"],
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def install_tailwind(self):
        return subprocess.run(
            ["npx", "tailwindcss", "init", "tailwind.config.cjs", "-p"],
            cwd=self.appdir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def install_packages(self, cwd=None):
        return subprocess.run(
            [self.package_manager, "install"],
            cwd=self.appdir if not cwd else cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def filter_installed_libraries(self, libraries: List[str]) -> List[str]:
        """Given a list of libraries, return the libraries that are
        installed in the app directory.

        Args:
            libraries (List[str]): List of libraries to check

        Returns:
            List[str]: List of libraries that are installed
        """
        p = subprocess.run(
            ["npm", "list"] + list(libraries) + ["--parseable", "--silent"],
            cwd=self.appdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = p.stdout.decode("utf-8").splitlines()
        # output consists of a list of paths to the installed package and
        # omits libraries that are not installed
        installed_libraries = [p for p in libraries for o in output if p in o]
        return installed_libraries

    def render_app_css(self):
        return jinja_env.get_template("app.css").render()

    def render_component_context(self):
        template = jinja_env.get_template("ComponentContext.svelte")
        components = self.get_all_components()
        frontend_components = self.get_all_frontend_components()

        # Get the libraries that the frontend components
        # depend on
        libraries = set([c.library for c in frontend_components])

        # Filter to only include components and frontend components
        # whose libraries are installed
        installed_libraries = self.filter_installed_libraries(libraries)
        components = [c for c in components if c.library in installed_libraries]
        frontend_components = [
            c for c in frontend_components if c.library in installed_libraries
        ]

        return template.render(
            components=components,
            frontend_components=frontend_components,
        )

    def render_component_wrapper(self, component: Type[Component]):
        # TODO: fix line breaks in Wrapper.svelte
        template = jinja_env.get_template("Wrapper.svelte")

        return template.render(
            import_style=component.wrapper_import_style,
            component_name=component.component_name,
            path=component.path,
            prop_names=component.prop_names,
            event_names=component.event_names,
            use_bindings=not issubclass(component, AutoComponent),
            prop_bindings=component.prop_bindings,
            slottable=component.slottable,
        )

    def render_components_init_py(self):
        return jinja_env.get_template("components.init.py").render()

    def render_constants_js(self):
        return jinja_env.get_template("constants.js").render()

    def render_example_component(self):
        return jinja_env.get_template("ExampleComponent.svelte").render()

    def render_example_py(self):
        return jinja_env.get_template("example.py").render()

    def render_gitingore(self):
        return jinja_env.get_template("gitignore.jinja").render()

    def render_installer_js(self):
        return jinja_env.get_template("installer/installer.js").render(
            name=self.appname,
        )

    def render_installer_package_json(self):
        return jinja_env.get_template("installer/package.json").render()

    def render_layout(self):
        return jinja_env.get_template("layout.svelte").render()

    def get_import_prefix(self):
        if self.is_user_appdir:
            # Use the @meerkat-ml/meerkat package instead of $lib
            # in a Meerkat generated app
            return NPM_PACKAGE
        return "$lib"

    def render_route(self, interface: "Interface"):
        template = jinja_env.get_template("page.svelte.jinja")

        # TODO: make this similar to render_root_route
        #       and use component.frontend_alias and component.alias
        return template.render(
            route=interface.id,
            title=interface.name,
            import_prefix=self.get_import_prefix(),
            components=list(sorted(interface.component.get_components())),
            queryparam=False,
        )

    def render_root_route(self):
        template = jinja_env.get_template("page.root.svelte.jinja")
        components = self.get_all_components()
        frontend_components = self.get_all_frontend_components()

        return template.render(
            title="Meerkat",
            import_prefix=self.get_import_prefix(),
            components=components,
            frontend_components=frontend_components,
        )

    def render_setup_py(self):
        return jinja_env.get_template("setup.py").render()

    def render_slug_route(self):
        return jinja_env.get_template("page.slug.svelte.jinja").render()

    def render_svelte_config(self):
        return jinja_env.get_template("svelte.config.js").render()

    def render_tailwind_config(self):
        return jinja_env.get_template("tailwind.config.cjs").render()

    def update_package_json(self):
        # Update the package.json
        #   Add "@meerkat-ml/meerkat" to dependencies
        #   Add "tailwindcss" "postcss" "autoprefixer" to devDependencies
        with open(f"{self.appdir}/package.json") as f:
            package = json.load(f)

        if "dependencies" not in package:
            package["dependencies"] = {}
        package["dependencies"][NPM_PACKAGE] = "latest"
        package["dependencies"]["@sveltejs/adapter-static"] = "latest"

        package["devDependencies"] = {
            **package["devDependencies"],
            **{
                "tailwindcss": "latest",
                "postcss": "latest",
                "autoprefixer": "latest",
            },
        }

        with open(f"{self.appdir}/package.json", "w") as f:
            json.dump(package, f)

    def write_all_component_wrappers(
        self,
        exclude_classes: Set[str] = {"AutoComponent", "Component"},
    ):
        from meerkat.interactive.startup import get_subclasses_recursive

        # Recursively find all subclasses of Component
        subclasses = get_subclasses_recursive(Component)
        for subclass in subclasses:
            # Use subclass.__name__ as the component name, instead of
            # subclass.component_name, because the latter is not guaranteed to be unique.
            component_name = subclass.__name__
            if component_name in exclude_classes:
                continue

            # Make a file for the component, inside a subdirectory for the namespace
            # e.g. src/lib/wrappers/__meerkat/Component.svelte
            self.write_component_wrapper(subclass)

    def write_app_css(self):
        self.write_file(f"{self.appdir}/src/app.css", self.render_app_css())

    def write_component_context(self):
        self.write_file(
            f"{self.appdir}/src/lib/ComponentContext.svelte",
            self.render_component_context(),
        )

    def write_component_wrapper(self, component: Type[Component]):
        cwd = f"{self.appdir}/src/lib/wrappers/__{component.namespace}"
        os.makedirs(cwd, exist_ok=True)
        self.write_file(
            f"{cwd}/{component.__name__}.svelte",
            self.render_component_wrapper(component),
        )

    def write_constants_js(self):
        self.write_file(
            f"{self.appdir}/src/lib/constants.js",
            self.render_constants_js(),
        )

    def write_dot_mk(self):
        self.write_file(f"{self.appdir}/.mk", "")

    def write_example_component(self):
        cwd = f"{self.appdir}/src/lib/components"
        os.makedirs(cwd, exist_ok=True)

        self.write_file(
            f"{cwd}/ExampleComponent.svelte",
            self.render_example_component(),
        )
        self.write_file(f"{cwd}/__init__.py", self.render_components_init_py())

    def write_example_py(self):
        self.write_file("example.py", self.render_example_py())

    def write_file(self, path: str, content: str):
        with open(path, "w") as f:
            f.write(content)

    def write_installer(self):
        os.makedirs("installer", exist_ok=True)
        self.write_file("installer/package.json", self.render_installer_package_json())
        self.write_file("installer/installer.js", self.render_installer_js())

    def write_gitignore(self):
        self.write_file(f"{self.appdir}/.gitignore", self.render_gitingore())

    def write_layout(self):
        self.write_file(
            f"{self.appdir}/src/routes/+layout.svelte",
            self.render_layout(),
        )
        self.write_file(
            f"{self.appdir}/src/routes/+layout.js",
            "export const prerender = true;",
        )

    def write_libdir(self):
        os.makedirs(f"{self.appdir}/src/lib", exist_ok=True)

    def write_setup_py(self):
        self.write_file("setup.py", self.render_setup_py())

    def write_slug_route(self):
        os.makedirs(f"{self.appdir}/src/routes/[slug]", exist_ok=True)
        self.write_file(
            f"{self.appdir}/src/routes/[slug]/+page.svelte",
            self.render_slug_route(),
        )

    def write_svelte_config(self):
        self.write_file(f"{self.appdir}/svelte.config.js", self.render_svelte_config())

    def write_tailwind_config(self):
        self.write_file(
            f"{self.appdir}/tailwind.config.cjs",
            self.render_tailwind_config(),
        )


# Create an instance that can be used by other modules
svelte_writer = SvelteWriter()
