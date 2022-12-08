import argparse
import inspect
import re

import black

parser = argparse.ArgumentParser()
parser.add_argument("filepath", help="The file to transpile")
args = parser.parse_args()

filepath = args.filepath
# filepath = "Button.svelte"
component_name = filepath.split("/")[-1].split(".")[0]

# Open the Svelte file, and parse all statements that have `export let` in them
# These are props that are passed in from the Python code
with open(filepath, "r") as f:
    lines = f.readlines()
    lines = " ".join([line.strip() for line in lines])
    matches = re.findall(
        r"export\s+let\s+(\w+)(?:\s*:\s*([\w\<\>]+))?(?:[;\s]*)(?:=\s*([^;]+)[;]+)?",
        lines,
    )

# mapping prop name to prop type and default value
props = {}
for (prop_name, prop_type, prop_default) in matches:
    if prop_default == "":
        prop_default = None
    try:
        prop_default, default_py_type = (
            eval(prop_default),
            type(eval(prop_default)).__name__,
        )
    except:  # noqa
        # Discard the default value if it's not a valid Python expression
        # TODO: currently this discards "a text with ;" due to the semicolon
        prop_default, default_py_type = None, None

    if prop_type == "string":
        py_type = "str"
    elif prop_type == "number":
        py_type = "float"
    elif prop_type == "boolean":
        py_type = "bool"
    elif prop_type == "Writable<string>":
        py_type = "str"
    elif prop_type == "Writable<number>":
        py_type = "float"
    elif prop_type == "Writable<boolean>":
        py_type = "bool"
    elif prop_type == "Writable":
        py_type = "Any"
    else:
        py_type = "Any"

    if default_py_type:
        py_type = default_py_type

    props[prop_name] = {
        "type": py_type,
        "default": prop_default,
    }


signature = inspect.Signature(
    parameters=[
        inspect.Parameter(name="self", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    + [
        inspect.Parameter(
            name=prop_name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=prop["default"],
            annotation=eval(prop["type"]),
        )
        for prop_name, prop in props.items()
    ]
)

# assign_statements = "\n".join([
#     f'        self.{prop_name} = make_store({prop_name})'
#     for prop_name in props
# ])
# print(assign_statements)

# Now generate a Python class that has the same props
# as the Svelte component
python_code = f"""from typing import Any
from meerkat.interactive import Component

class {component_name}(Component):
    name = "{component_name}"
    def __init__{signature}:
        super().__init__({', '.join(props.keys())})
"""
# Run black on the generated code
python_code = black.format_str(python_code, mode=black.FileMode())

# Write to a file
with open(f"{component_name}.py", "w") as f:
    f.write(python_code)
