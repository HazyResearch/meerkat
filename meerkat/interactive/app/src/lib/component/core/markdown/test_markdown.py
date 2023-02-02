import meerkat as mk

title = mk.gui.core.Title(body="Title")
header = mk.gui.core.Header(body="Header")
subheader = mk.gui.core.Subheader(body="Subheader")
caption = mk.gui.core.Caption(body="Caption")

markdown = mk.gui.core.Markdown(
    body="""
# Hello world
This is a markdown component.
We can show bold text like this: **bold**.
We can show italic text like this: *italic*.
We can show code like this: `code`.
We can show a link like this: [link](https://meerkat.ml).
We can show a list like this:
- item 1
- item 2
- item 3

We can show a table like this:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| item 1   | item 2   | item 3   |
| item 3   | item 4   | item 5   |

Add a Python code block:
```python
import meerkat as mk
```

Add a JavaScript code block:
```javascript
console.log("Hello world");
```
""",
    classes="prose prose-sm",
    breaks=True,
)

component = mk.gui.html.div(slots=[
    mk.gui.html.div(slots=[title, header, subheader, caption]),
    mk.gui.html.div(slots=[markdown], classes="mt-8"),
])

page = mk.gui.Page(component=component, id="markdown")
page.launch()
