# flake8: noqa
import argparse
import importlib

import meerkat as mk

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
doc_data = utils.load_fairytale_qa_doc(
    doc_path=args.data_dir + "FairytaleQA_Dataset/split_by_origin"
)
story_doc_df = doc_data.story_doc_df

##### Build the interface #####

# Start the Meerkat GUI server
gui_info = mk.gui.start(shareable=False)

# Wrap the DataFrame with a `Reference`
# This makes it a starting point for the interface
story_doc_df = mk.gui.Reference(story_doc_df)

# Make a Choice component: this allows us to choose a document
doc_id_choice = mk.gui.Choice(
    value=story_doc_df["id"][0],  # the default value is the first document id
    choices=list(story_doc_df["id"]),  # choices are the document ids
)
# ...access the chosen document id with `choice.value`
# this is a `Store` object whose value can be accessed with `<store>.value`
# Stores are a way to keep track of state that is changing as the user interacts
# with the frontend. Their value will automatically be synced by Meerkat
# with the frontend.

# Hacky way to get the index of the chosen document id
# We're adding support for primary keys in Meerkat, so this will be
# cleaned up in a couple of weeks
_index = dict(zip(story_doc_df["id"], range(len(story_doc_df["id"]))))


@mk.gui.reactive
def choose_doc(doc_id):
    """
    This is an interactive operation (decorate with `@mk.gui.interface_op`)
    that is retriggered whenever the user changes the values of the arguments.

    Here, when the user chooses something in the `Choice` component
    i.e. changes the value of `doc_id_choice.value`, this function will
    - get the index of the chosen `doc_id`
    - return the corresponding row from the `story_doc_df` DataFrame,
    for column `section_df`

    Important: this code needs to be inside a function, e.g. if you
    copy and paste this code outside of the function, it won't work.
    This is because the `@mk.gui.interface_op` decorator does some
    work to unwrap the `doc_id` Store.
    """
    # Get the document index
    doc_idx = _index[doc_id]
    return story_doc_df["section_df"][doc_idx]


# Create the Document component
document = mk.gui.Document(
    df=choose_doc(doc_id_choice.value),  # sections of the chosen document
    text_column="text",  # the column that contains document text
)

# Launch the interface
mk.gui.Interface(
    components=[doc_id_choice, document],
).launch()
