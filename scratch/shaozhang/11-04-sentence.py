# flake8: noqa
import argparse
import importlib

import numpy as np

import meerkat as mk

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="/Users/zhangshao/program/FairytaleQA_Dataset/",
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
story_sentence_df["id"] = np.arange(len(story_sentence_df))
story_sentence_df.drop("sentence_id")

# Create an empty label column 
# (-1 means no label, 0 means negative, 1 means positive, 2 means unknown)
story_sentence_df["label"] = np.zeros(len(story_sentence_df)) - 1

##### Build the interface #####

# Start the Meerkat GUI server
gui_info = mk.gui.start(shareable=False)

# Wrap the DataFrame with a `Reference`
# This makes it a starting point for the interface
story_sentence_df_ref = mk.gui.Reference(story_sentence_df)

# Make a Choice component: this allows us to choose a document
doc_id_choice = mk.gui.Choice(
    value=story_sentence_df["doc_id"][0],  # the default value is the first document id
    choices=list(story_sentence_df["doc_id"].unique()),  # choices are the document ids
)
# ...access the chosen document id with `choice.value`
# this is a `Store` object whose value can be accessed with `<store>.value`
# Stores are a way to keep track of state that is changing as the user interacts
# with the frontend. Their value will automatically be synced by Meerkat
# with the frontend.

# Hacky way to get the index of the chosen document id
# We're adding support for primary keys in Meerkat, so this will be
# cleaned up in a couple of weeks
_index = dict(zip(story_sentence_df["doc_id"].unique(), range(len(story_sentence_df["doc_id"].unique()))))
_sentence_index = mk.ArrayColumn(story_sentence_df['doc_id'])[None, :] == mk.ArrayColumn(story_sentence_df['doc_id'].unique())[:, None]
_get_sentences = lambda idx: np.where(_sentence_index[idx])[0]

@mk.gui.interface_op
def choose_sentences(sentence_df: mk.DataFrame, doc_id: str):
    """
    This is an interactive operation (decorate with `@mk.gui.interface_op`)
    that is retriggered whenever the user changes the values of the arguments.

    Here, when the user chooses something in the `Choice` component
    i.e. changes the value of `doc_id_choice.value`, this function will
    - get the index of the chosen `doc_id`
    - return the corresponding row from the `story_sentence_df` DataFrame,
    for column `section_df`

    Important: this code needs to be inside a function, e.g. if you
    copy and paste this code outside of the function, it won't work.
    This is because the `@mk.gui.interface_op` decorator does some
    work to unwrap the `doc_id` Store.
    """
    # Get the document index
    doc_idx = _index[doc_id]
    sentence_indices = _get_sentences(doc_idx)
    return sentence_df[sentence_indices]


"""
Code snippet to create an editing target (EditTarget), which we will send to the frontend to tell it
which dataframe to send the `edit` API call request to. This will let us send user labels to the right place.
"""
# Create an EditTarget component
target = mk.gui.EditTarget(
    target=story_sentence_df_ref, 
    target_id_column="id", 
    source_id_column="id",
)

"""
Code snippet that creates the Document labeling interface.
We pass it the (i) dataframe, (ii) the columns that correspond to the text, section index and labels
and (iii) the EditTarget component that we created above.
"""
# Create the Document component
document = mk.gui.Document(
    df=choose_sentences(story_sentence_df_ref, doc_id_choice.value),  # sections of the chosen document
    text_column="text",  # the column that contains document text
    paragraph_column="section",  # the column that contains the paragraph index
    label_column="label",  # the column that contains the label
    edit_target=target,  # the edit target
)

"""
Code snippet that autosaves labels to a jsonl file, whenever the user updates any label.

TODO: it is very easy to also just load these back in to continue a labeling session as well!
"""
@mk.gui.interface_op
def save_labels(df: mk.DataFrame):
    """
    Save the labels to a file.
    """
    df[['label']].to_jsonl("labels.jsonl")

# Runs the save_labels function whenever the `story_sentence_df` updates
save_labels(story_sentence_df_ref)

# Launch the interface
mk.gui.Interface(
    components=[doc_id_choice, document],
).launch()
