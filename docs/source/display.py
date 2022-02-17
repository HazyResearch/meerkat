import meerkat as mk


def display_dp(dp: mk.DataPanel, name: str):
    body_html = dp._repr_html_()
    css = open("source/html/display/datapanel.css", "r").read()
    body_html = body_html.replace("\n", f"\n <style> {css} </style>", 1)
    open(f"source/html/display/{name}.html", "w").write(body_html)
    return dp
