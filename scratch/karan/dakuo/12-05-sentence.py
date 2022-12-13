# flake8: noqa
import argparse
import importlib

import numpy as np

import meerkat as mk
from meerkat.interactive.app.src.lib.layouts import RowLayout

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="/Users/krandiash/Desktop/workspace/projects/datasci/data/FairytaleQA_Dataset/",
)
args = parser.parse_args()

# Load functions from the script '11-04-clean-data.py'
utils = importlib.import_module("11-04-clean-data")

##### Load the data #####
try:
    story_sentence_df = mk.DataFrame.read("story_sentence_processed_df.mk")
except:
    print("Creating story_sentence_processed_df.mk, will take around 3 minutes...")
    utils.create_story_sentence_df(args.data_dir)
    story_sentence_df = mk.DataFrame.read("story_sentence_processed_df.mk")

doc_data = utils.load_fairytale_qa_doc(
    doc_path=args.data_dir + "FairytaleQA_Dataset/split_by_origin"
)

# Make sure all the ID columns make sense
story_sentence_df["doc_id"] = story_sentence_df["sentence_id"]
story_sentence_df.create_primary_key("id")
story_sentence_df.drop("sentence_id")

# Create an empty label column
# (-1 means no label, 0 means negative, 1 means positive, 2 means unknown)
story_sentence_df["label"] = np.zeros(len(story_sentence_df)) - 1

# Mappings to go from a document id to the sentences in that document
_index = dict(
    zip(
        story_sentence_df["doc_id"].unique(),
        range(len(story_sentence_df["doc_id"].unique())),
    )
)
_sentence_index = (
    mk.ArrayColumn(story_sentence_df["doc_id"])[None, :]
    == mk.ArrayColumn(story_sentence_df["doc_id"].unique())[:, None]
)
_get_sentences = lambda idx: np.where(_sentence_index[idx])[0]

##### Build the interface #####

# Start the Meerkat GUI server
gui_info = mk.gui.start(shareable=False)

# Make a Choice component: this allows us to choose a document
doc_id_choice = mk.gui.Choice(
    value=story_sentence_df["doc_id"][0],  # the default value is the first document id
    choices=list(story_sentence_df["doc_id"].unique()),  # choices are the document ids
    title="Choose a document",
)
# ...access the chosen document id with `choice.value`
# This is a `Store` object which can be used as if it were a normal Python variable
# Stores are a way to keep track of state that is changing as the user interacts
# with the frontend. Their value will automatically be synced by Meerkat
# with the frontend.


@mk.gui.reactive
def choose_sentences(sentence_df: mk.DataFrame, doc_id: str):
    """
    This is a reactive function (decorated with `@mk.gui.reactive`)
    that can be retriggered whenever the user changes the values of the arguments.

    To execute this function reactively, place it inside a
    `mk.gui.react()` context manager i.e.

    with mk.gui.react():
        document_df = choose_sentences(sentence_df, doc_id)

    When the user chooses something in the `Choice` component
    i.e. changes the value of `doc_id_choice.value`, this function will
    - get the index of the chosen `doc_id`
    - return the corresponding row from the `story_sentence_df` DataFrame,
    for column `section_df`
    """
    # Get the document index
    doc_idx = _index[doc_id]
    sentence_indices = _get_sentences(doc_idx)
    return sentence_df[sentence_indices]


with mk.gui.react():
    # Use the Choice component to choose a document and create a DataFrame for it
    document_df = choose_sentences(story_sentence_df, doc_id_choice.value)

"""
Code snippet that creates the Document labeling interface.
We pass it the 
    (i) DataFrame, 
    (ii) the columns that correspond to the text, section index, labels and ids,
    (iii) an Endpoint that can be used to store the labels sent by the frontend
"""
from meerkat.interactive.api.routers.dataframe import edit

# Create the Document component
document = mk.gui.Document(
    df=document_df,  # sections of the chosen document
    text_column="text",  # the column that contains document text
    paragraph_column="section",  # the column that contains the paragraph index
    label_column="label",  # the column that contains the label
    id_column="id",  # the column that contains the sentence id
    on_sentence_label=edit.partial(
        df=story_sentence_df, column="label", id_column="id"
    ),
)


@mk.gui.reactive
def save_labels(df: mk.DataFrame):
    """Save the labels to a file."""
    df["id", "label"].to_jsonl("labels.jsonl")


with mk.gui.react():
    # Autosave labels whenever the user updates `story_sentence_df` with a new label
    save_labels(story_sentence_df)

# This will not work as a reactive function anymore
# save_labels(story_sentence_df)

# Launch the interface
mk.gui.Interface(
    component=RowLayout(components=[doc_id_choice, document]),
).launch()


# Layouts are also Components: everything is a Component for GUI stuff
# and then you just pass it to Interface
# Way to layout components
# mk.gui.Flex(components=[
#     doc_id_choice,
#     document
# ], classes="...") # TAILWIND CSS
# equivalent to 
# <div class="flex flex-row">


# Flex, Grid, Div, RowLayout, ColumnLayout
# Div(components=[Flex(...), Flex(...)])
