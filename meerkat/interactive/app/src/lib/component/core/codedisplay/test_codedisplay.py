from meerkat.interactive import Page
from meerkat.interactive.app.src.lib.component.codedisplay import CodeDisplay

code_py = CodeDisplay(
    data="""\
from meerkat.interactive import Page, endpoint
from meerkat.interactive.app.src.lib.component.codedisplay import CodeDisplay
"""
)

page = Page(component=code_py, id="codedisplay")
page.launch()
