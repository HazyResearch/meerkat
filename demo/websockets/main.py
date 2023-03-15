import meerkat as mk


page = mk.gui.Page(mk.gui.html.div("Hello world!"), id="page")
page.launch()
