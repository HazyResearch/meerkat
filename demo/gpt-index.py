import os

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

import meerkat as mk

dir_name = mk.gui.Store(None)


@mk.gui.react()
def load_index(dir_name: str) -> GPTSimpleVectorIndex:
    if not dir_name:
        return None

    if os.path.exists(os.path.join(dir_name, f"{dir_name}.json")):
        # load the index after savaing (maybe)
        return GPTSimpleVectorIndex.load_from_disk(f"{dir_name}.json")
    else:
        raise FileNotFoundError("GPT index.json file not found.")

    # reader for docs
    documents = SimpleDirectoryReader(dir_name).load_data()

    # build the index
    index = GPTSimpleVectorIndex(documents)

    # save the index somewhere (maybe)
    index.save_to_disk("file_path.json")

    return index


@mk.gui.endpoint
def query_gpt_index(index: GPTSimpleVectorIndex, query: str):
    if index:
        return "Index not created. Please use the picker to choose a folder."
    return index.query(query)


index = load_index(dir_name=dir_name)

picker = mk.gui.SomeComponent(
    value=dir_name,
)

querier = mk.gui.GPTIndexQueryComponent(
    # This "special" component sends an event called "query"
    # that contains a payload with a key called `query` that contains
    # the string query
    on_query=query_gpt_index.partial(index=index),
)

interface = mk.gui.Interface(
    component=mk.gui.RowLayout(slots=[picker, querier]),
    id="gpt-index",
)
interface.launch()
