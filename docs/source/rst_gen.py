"""Functions to generate certain RST files."""
import os
import pathlib

import meerkat as mk

_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))


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
        lines.append("   :toctree: generated")
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


if __name__ == "__main__":
    generate_inbuilt_reactive_fns()
