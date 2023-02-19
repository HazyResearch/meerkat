"""Functions to generate certain RST files."""
import os
import pathlib
from collections import defaultdict
from typing import List, Union

import pandas as pd

import meerkat as mk

_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))


def _replace_contents_in_file(fpath, *, key: str, contents: Union[str, List[str]]):
    """Replace contents of a file that are between the delimiters.

    Delimiters should be Markdown style comments formatted with the key.
    For example:

        <!---Start: mk-store-reactive-operators-->
        <!---End: mk-store-reactive-operators-->
    """
    if isinstance(contents, str):
        contents = [contents]
    contents = [line + "\n" if not line.endswith("\n") else line for line in contents]
    if not contents[-1].endswith("\n"):
        contents += ["\n"]

    start_delimiter = f"<!---autogen-start: {key}-->"
    end_delimiter = f"<!---autogen-end: {key}-->"

    # Read in the file
    with open(fpath, "r") as file:
        lines = file.readlines()

    start_indices = [
        idx for idx, line in enumerate(lines) if line.strip() == start_delimiter
    ]
    end_indices = [
        idx for idx, line in enumerate(lines) if line.strip() == end_delimiter
    ]
    if len(start_indices) != len(end_indices):
        raise ValueError(f"Number of start and end delimiters do not match in {fpath}.")
    if len(start_indices) == 0:
        raise ValueError(f"No start and end delimiters found in {fpath}.")

    # Replace the content between the delimiters.
    brackets = lines[: start_indices[0] + 1] + contents
    for end_idx, next_start_idx in zip(end_indices, start_indices[1:]):
        brackets.extend(lines[end_idx : next_start_idx + 1])
        brackets.extend(contents)
    brackets.extend(lines[end_indices[-1] :])

    # Write the file out again
    with open(fpath, "w") as file:
        file.writelines(brackets)


def generate_inbuilt_reactive_fns():
    """Generate the inbuilt reactive functions RST file."""
    fpath = _DIR / "guide" / "reactive" / "inbuilts.rst"

    _REACTIVE_FNS = {
        "General": [
            "all",
            "any",
            "bool",
            "complex",
            "float",
            "hex",
            "int",
            "len",
            "oct",
            "str",
            "list",
            "tuple",
            "sum",
            "dict",
            "set",
            "range",
            "abs",
        ],
        "Boolean Operations": ["cand", "cnot", "cor"],
        "DataFrame Operations": [
            "concat",
            "merge",
            "sort",
            "sample",
            "shuffle",
            "groupby",
            "clusterby",
            "aggregate",
            "explainby",
        ],
    }

    lines = [
        ".. _reactivity_inbuilts:",
        "",
        "Reactive Functions in Meerkat",
        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        "Meerkat provides some reactive functions out of the box.",
    ]

    # Add the reactive functions to the lines.
    for category, fns in _REACTIVE_FNS.items():
        fns = sorted(fns)
        lines.append("")
        lines.append(category)
        lines.append("-" * len(category))
        lines.append("")
        lines.append(".. autosummary::")
        lines.append("   :toctree: ../../apidocs/generated")
        lines.append("   :nosignatures:")
        lines.append("")
        for fn in fns:
            assert hasattr(mk, fn), f"mk.{fn} is not a function in Meerkat."
            mk_fn = getattr(mk, fn)
            assert (
                hasattr(mk_fn, "__wrapper__") and mk_fn.__wrapper__ == "reactive"
            ), f"mk.{fn} is not a reactive function."
            lines.append(f"   meerkat.{fn}")

    with open(fpath, "w") as f:
        for line in lines:
            f.write(line + "\n")


def generate_store_operators():
    """Generate table of store operators that are reactive."""
    fpath = _DIR / "guide" / "store" / "user-guide.md"
    operators = [
        # Arthmetic
        "Addition, +, x + y",
        "Subtraction, -, x - y",
        "Multiplication, *, x * y",
        "Division, /, x / y",
        "Floor Division, //, x // y",
        "Modulo, %, x % y",
        "Exponentiation, **, x ** y",
        # Assignment
        "Add & Assign, +=, x+=1",
        "Subtract & Assign, -=, x-=1",
        "Multiply & Assign, *=, x*=1",
        "Divide & Assign, /=, x/=1",
        "Floor Divide & Assign, //=, x//=1",
        "Modulo & Assign, %=, x%=1",
        "Exponentiate & Assign, **=, x**=1",
        "Power & Assign, **=, x**=1",
        "Bitwise Left Shift & Assign, <<=, x<<=1",
        "Bitwise Right Shift & Assign, >>=, x>>=1",
        "Bitwise AND & Assign, &=, x&=1",
        "Bitwise XOR & Assign, ^=, x^=1",
        "Bitwise OR & Assign, |=, x|=1",
        # Bitwise
        "Bitwise Left Shift, <<, x << y",
        "Bitwise Right Shift, >>, x >> y",
        "Bitwise AND, &, x & y",
        "Bitwise XOR, ^, x ^ y",
        "Bitwise OR, |, x | y",
        "Bitwise Inversion, ~, ~x",
        # Comparison
        "Less Than, <, x < y",
        "Less Than or Equal, <=, x <= y",
        "Equal, ==, x == y",
        "Not Equal, !=, x != y",
        "Greater Than, >, x > y",
        "Greater Than or Equal, >=, x >= y",
        # Get item
        "Get Item, [key], x[0]",
        "Get Slice, [start:stop], x[0:10]",
    ]

    content = defaultdict(list)
    for operator in operators:
        name, symbol, example = [x.strip() for x in operator.split(",")]
        content["name"].append(name)
        content["symbol"].append(symbol)
        content["example"].append(example)

    df = pd.DataFrame(content)
    content_markdown = df.to_markdown(index=False)
    _replace_contents_in_file(
        fpath, key="mk-store-reactive-operators", contents=content_markdown
    )


if __name__ == "__main__":
    generate_inbuilt_reactive_fns()
    generate_store_operators()
