import json

import meerkat as mk

component = mk.gui.core.Json(
    body=json.loads("""\
{
    "name": "John Doe",
    "age": 43,
    "children": [
        {
            "name": "Sally",
            "age": 13
        },
        {
            "name": "Billy",
            "age": 8
        }
    ]
}
"""),
    padding=2,
)

page = mk.gui.Page(component=component, id="json")
page.launch()
