import os

import rich
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

import meerkat as mk
from meerkat.interactive.app.src.lib.component.fileupload import FileUpload


@mk.gui.react()
def rprint(*args):
    """Helper function that reactively prints its arguments to the console."""
    rich.print(*args)


# Start off with a default directory
dir = mk.gui.Store("/Users/krandiash/Desktop/workspace/projects/gpt_index_data/")
savefilename = "gpt_index"


@mk.gui.react()
def load_index(dir: str, savefilename: str) -> GPTSimpleVectorIndex:
    if not dir or not os.path.exists(dir):
        return None

    if os.path.exists(os.path.join(dir, f"{savefilename}.json")):
        # load the index after savaing (maybe)
        return GPTSimpleVectorIndex.load_from_disk(
            os.path.join(dir, f"{savefilename}.json")
        )

    # reader for docs
    documents = SimpleDirectoryReader(dir).load_data()

    # build the index
    index = GPTSimpleVectorIndex(documents)

    # save the index somewhere (maybe)
    index.save_to_disk(os.path.join(dir, f"{savefilename}.json"))

    return index


# Create the index, and assign it to a variable.
index = load_index(dir=dir, savefilename=savefilename)

# Create a variable that will be used to store the response from the index
# for the last query. We will display this in the UI.
last_response = mk.gui.Store("The response will appear here.")


@mk.gui.endpoint
def query_gpt_index(index: GPTSimpleVectorIndex, query: str):
    """
    A function that given an index and query, will return the response from the index.
    """
    if not index:
        return "Index not created. Please use the picker to choose a folder."
    response = index.query(query)
    last_response.set(response.response)
    rich.print(response)
    return response


# FileUpload component, which can be used to upload files.
# fileupload_component = FileUpload()

# Create a Store that will hold the query.
query = mk.gui.Store("")
query_component = mk.gui.Textbox(text=query, title="Question")

# Create a Store that will hold the dir.
dir_component = mk.gui.Textbox(text=dir, title="Directory")

# Create a button that will call the query_gpt_index endpoint when clicked.
button = mk.gui.Button(
    title="Query GPT-Index",
    on_click=query_gpt_index.partial(index=index, query=query),
)

text = mk.gui.html.div(
    slots=[mk.gui.html.p(slots=last_response, classes="font-mono whitespace-pre-wrap")],
    classes="flex flex-col items-center justify-center h-full mt-4 bg-violet-200",
)

# Print the values of the variables to the console, so we can see them.
# This will reprint them whenever any of the variables change.
rprint("\n", "Query:", query, "\n", "Dir:", dir, "\n", "Index:", index, "\n")


interface = mk.gui.Interface(
    component=mk.gui.RowLayout(
        slots=[
            # fileupload_component,
            dir_component,
            query_component,
            button,
            text,
        ]
    ),
    id="gpt-index",
)
interface.launch()
