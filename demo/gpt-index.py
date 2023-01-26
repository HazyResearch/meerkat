import os

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

import meerkat as mk
from meerkat.interactive.app.src.lib.component.fileupload import FileUpload

dir = mk.gui.Store(None)
query = mk.gui.Store("")
savefilename = "gpt_index"


@mk.gui.react()
def load_index(dir: str, savefilename: str) -> GPTSimpleVectorIndex:
    print(
        f"Executing load_index with dir={type(dir)} and savefilename={type(savefilename)}"
    )
    if not dir:
        return None

    if os.path.exists(os.path.join(dir, f"{savefilename}.json")):
        # load the index after savaing (maybe)
        return GPTSimpleVectorIndex.load_from_disk(f"{savefilename}.json")

    # reader for docs
    documents = SimpleDirectoryReader(dir).load_data()

    # build the index
    index = GPTSimpleVectorIndex(documents)

    # save the index somewhere (maybe)
    index.save_to_disk(os.path.join(dir, f"{savefilename}.json"))

    print("Index created.")

    return index


@mk.gui.endpoint
def query_gpt_index(index: GPTSimpleVectorIndex, query: str):
    if index:
        return "Index not created. Please use the picker to choose a folder."
    return index.query(query)


index = load_index(dir=dir, savefilename=savefilename)

# Make a FileUpload component to select a directory
fileupload_component = FileUpload(value=dir)
query_component = mk.gui.Textbox(text=query)

button = mk.gui.Button(
    title="Query GPT-Index",
    on_click=query_gpt_index.partial(index=index, query=query),
)

interface = mk.gui.Interface(
    component=mk.gui.RowLayout(
        slots=[
            fileupload_component,
            query_component,
            button,
        ]
    ),
    id="gpt-index",
)
interface.launch()
