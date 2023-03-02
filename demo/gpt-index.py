import os

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

import meerkat as mk
from meerkat.interactive import (
    Button,
    Caption,
    Header,
    Page,
    Store,
    Subheader,
    Textbox,
    endpoint,
    html,
    print,
    reactive,
)


# Decorate with @reactive so that this fn is reactive.
# When its inputs change, it will be re-run.
@reactive
def load_index(dir: str, savefilename: str) -> GPTSimpleVectorIndex:
    """Function that loads the GPTSimpleVectorIndex from disk if it exists.
    Otherwise it builds the index and saves it to disk.

    Args:
        dir (str): The directory where the index is saved.
        savefilename (str): The name of the file where the index is saved.

    Returns:
        GPTSimpleVectorIndex: The index.
    """
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


# Start off with a default directory.
#   Make `dir` a `Store` to create a variable that can be used in reactive functions.
#   A `Store` is a thin wrapper that behaves like the object it wraps.
dir = mk.Store("/Users/krandiash/Desktop/workspace/projects/gpt_index_data/")

# Create the index.
# Pass `dir` directly as if it were a `str`.
index = load_index(dir=dir, savefilename="gpt_index")

# Create a variable that will be used to store the response from the index
# for the last query. We will display this in the UI.
last_response = mk.Store("The response will appear here.")


@endpoint()
def query_gpt_index(
    index: GPTSimpleVectorIndex,
    query: str,
    last_response: Store,
):
    """Given an index and query, return a response from the index.

    Args:
        index (GPTSimpleVectorIndex): The index.
        query (str): The query.
        last_response (Store): The store that will hold the last response.
    """
    if not index:
        last_response.set(
            "Index not created. Please use the picker to choose a folder."
        )
        return

    response = index.query(query)

    # Stores can only be set inside endpoints in order to trigger the reactive
    # functions that depend on them! The special `set` method helps with this.
    last_response.set(response.response)
    return response


# Create a Textbox for the user to provide a query.
query_component = Textbox()

# Create a Textbox for the user to provide a directory.
dir_component = Textbox(dir)

# Create a button that will call the `query_gpt_index` endpoint when clicked.
button = Button(
    title="Query GPT-Index",
    on_click=query_gpt_index.partial(
        index=index,
        query=query_component.text,
        last_response=last_response,
    ),
)

# Display the response from the index.
# Use HTML to display the response from the index nicely.
text = html.div(
    html.p(last_response, classes="font-mono whitespace-pre-wrap"),
    classes="flex flex-col items-center justify-center h-full mt-4 bg-violet-200 rounded-sm p-2",  # noqa: E501
)

# Print the values of the variables to the console, so we can see them.
# This will reprint them whenever any of the variables change.
print(
    "\n", "Query:", query_component.text, "\n", "Dir:", dir, "\n", "Index:", index, "\n"
)

page = Page(
    # Layout the Interface components one row each, and launch the page.
    html.flexcol(
        [
            # fileupload_component,
            Header("GPT-Index Demo"),
            Subheader("Directory Path"),
            Caption(
                "It should contain a few files to build the GPT-Index. "
                f"Defaults to {dir}."
            ),
            dir_component,
            Subheader("Query"),
            Caption("Ask a question!"),
            query_component,
            button,
            text,
        ]
    ),
    id="gpt-index",
)
page.launch()
