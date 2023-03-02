from meerkat.interactive.app.src.lib.component.abstract import Component


class Code(Component):
    body: str
    theme: str = "okaidia"
    background: str = "bg-slate-800"
    language: str = "python"
    classes: str = ""
