import meerkat as mk
from meerkat.interactive.svelte import svelte_writer


def test_index_js():
    # Read the index.js file
    path = mk.__path__[0] + "/interactive/app/src/lib/index.js"

    with open(path, "r") as f:
        index_js = f.read()

    # Extract all the export statements
    export_statements = [
        line for line in index_js.splitlines() if line.startswith("export")
    ]

    # Keep only the statements for .svelte files
    export_statements = [line for line in export_statements if ".svelte" in line]

    # Assert that all statements look like:
    # export { default as Button } from './component/button/Button.svelte';
    for line in export_statements:
        assert line.startswith("export { default as ")

    # Extract the names of the components from "default as XXX"
    exported_components = sorted(
        [line.split("default as ")[-1].split(" }")[0] for line in export_statements]
    )

    # Remove the `Page` and `Meerkat` components
    exported_components = [
        component
        for component in exported_components
        if component not in ["Page", "Meerkat"]
    ]

    # Get the list of all components defined in Python for Meerkat
    py_components = svelte_writer.get_all_components()

    # Keep only the components that have library @meerkat-ml/meerkat
    py_components = sorted(
        [
            component.component_name
            for component in py_components
            if component.library == "@meerkat-ml/meerkat"
        ]
    )

    # Assert that the list of components in Python is the same as the list of
    # components exported in index.js
    assert set(py_components) == set(exported_components), (
        "The list of components exported in app/src/lib/index.js is not the same as the list "
        "of components defined in Python. Make sure to export all components in "
        "index.js and to define all components in Python (with @meerkat-ml/meerkat "
        "library)."
    )
