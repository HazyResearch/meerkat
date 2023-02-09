import abc
from string import Template
from textwrap import dedent, indent
from typing import Any, Callable, Dict

FuncType = Callable[..., Any]

GLOBAL_DOCS = {}


def doc(source: Dict[str, str] = None, **kwargs) -> Callable[[FuncType], FuncType]:
    """
    A decorator take docstring templates, concatenate them and perform string
    substitution on it.
    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.
    Parameters
    ----------
    *docstrings : None, str, or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.

    References:
        This decorator is adapted from `pandas.util._decorators.doc`
    """
    if source is None:
        source = GLOBAL_DOCS

    def decorator(decorated: FuncType) -> FuncType:
        docstring = decorated.__doc__

        if docstring is None:
            docstring = ""

        fixed_source = {
            key: (
                value.fix_indentation(docstring)
                if isinstance(value, DocComponent)
                else value
            )
            for key, value in source.items()
        }

        docstring = Template(docstring).safe_substitute(**fixed_source)
        docstring = Template(docstring).safe_substitute(**kwargs)

        decorated.__doc__ = docstring
        return decorated

    return decorator


class DocComponent(abc.ABC):
    def __init__(self, text: str):
        self.text = text

    @abc.abstractmethod
    def fix_indentation(self, docstring: str) -> str:
        raise NotImplementedError()


class DescriptionSection(DocComponent):
    def fix_indentation(self, docstring: str) -> str:
        # get common leading whitespace from docstring ignoring first line
        lines = docstring.splitlines()
        leading_whitespace = min(
            len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()
        )

        prefix = leading_whitespace * " "
        text = indent(dedent(self.text), prefix)

        return text


class Arg(DocComponent):
    def fix_indentation(self, docstring: str) -> str:
        # get common leading whitespace from docstring ignoring first line
        lines = docstring.splitlines()
        leading_whitespace = min(
            len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()
        )

        prefix = (leading_whitespace + 4) * " "
        text = indent(dedent(self.text), prefix)

        return text


class ArgDescription(DocComponent):
    def fix_indentation(self, docstring: str) -> str:
        # get common leading whitespace from docstring
        lines = docstring.splitlines()
        leading_whitespace = min(
            len(line) - len(line.lstrip()) for line in lines if line.strip()
        )

        prefix = (leading_whitespace + 4) * " "
        text = indent(dedent(self.text), prefix)

        # remove prefix from first line
        text = text[len(prefix) :]

        return text
