import meerkat as mk

code_py = mk.gui.core.Code(
    body="""\
from meerkat.interactive import Page, endpoint
from meerkat.interactive.app.src.lib.component.codedisplay import CodeDisplay
"""
)

page = mk.gui.Page(component=code_py, id="code")
page.launch()
