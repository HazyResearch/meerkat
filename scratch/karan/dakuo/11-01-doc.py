# flake8: noqa
import argparse
from glob import glob
from types import SimpleNamespace

import numpy as np

import meerkat as mk

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="/Users/krandiash/Desktop/workspace/projects/datasci/data/FairytaleQA_Dataset/",
)
args = parser.parse_args()


##### Load the data #####

# Set up your paths
DATA_DIR = args.data_dir
DOC_PATH = DATA_DIR + "FairytaleQA_Dataset/split_by_origin"
SENTENCE_PATH = DATA_DIR + "FairytaleQA_Dataset_Sentence_Split/split_by_origin"


def load_doc_csv(row):
    """
    Loader for a single document csv file. This function takes a row of the doc_dp DataPanel
    and returns a DataPanel with the contents of the csv file.

    row: a row from the doc_dp DataPanel
    """
    # Load the csv file from disk as a DataPanel
    doc_data = mk.DataPanel.from_csv(row["path"])  # access the 'path'
    # Just add a doc_id column, so we know which document this CSV came from
    doc_data["doc_id"] = [row["id"]] * len(doc_data)  # access the 'id'
    return doc_data


def load_fairytale_qa_doc():
    # Create a Document DataPanel
    doc_dp = mk.DataPanel(
        {"path": glob(DOC_PATH + "/*/*.csv")}
    )  # put in a column called 'path' with all the paths to the csv files

    # Create some useful columns
    doc_dp["source"] = doc_dp["path"].apply(
        lambda x: x.split("/")[-2]
    )  # the source of the document e.g. "anderson-fairybook"
    doc_dp["name"] = doc_dp["path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )  # the name of the document e.g. "brave-tin-soldier-story.csv"
    doc_dp["id"] = (
        doc_dp["source"] + "/" + doc_dp["name"]
    )  # the id of the document e.g. "anderson-fairybook/brave-tin-soldier-story"

    doc_dp["type"] = (
        doc_dp["name"].str.split("-").str.get(-1)
    )  # the type of document among [story, questions] -- here we use pandas string methods directly on the column

    doc_dp["section_dp"] = doc_dp.map(
        load_doc_csv
    )  # .map will run the function on each line of the DataPanel and put the result in a new column called 'section_dp'

    # Split the doc_dp DataPanel into two DataPanels, one for stories and one for questions
    story_doc_dp = doc_dp[doc_dp["type"] == "story"]
    question_doc_dp = doc_dp[doc_dp["type"] == "questions"]

    # Magic: we can now access all the DataPanels inside story_doc_dp, concatenate them and get out a single DataPanel
    story_doc_sections_dp = mk.concat(story_doc_dp["section_dp"])

    # Similarly, we can do this for the question DataPanels
    try:
        # This doesn't work, because a couple of the question DataPanels have different columns
        question_doc_sections_dp = mk.concat(question_doc_dp["section_dp"])
    except:
        # We can check which rows have different columns
        has_extra_column = np.where(
            question_doc_dp.map(lambda x: len(x["section_dp"].columns)) == 16
        )[0]
        print(
            "These rows have an extra column, while all the others have 15 columns",
            has_extra_column,
        )  # 47, 109
        extra_column_name = set(question_doc_dp[47]["section_dp"].columns) - set(
            question_doc_dp[0]["section_dp"].columns
        )
        print("The extra column is called", extra_column_name)  # called "comments"
        # We can fix this by dropping the extra column if it exists
        question_doc_dp["section_dp"] = question_doc_dp["section_dp"].map(
            lambda x: x.drop(extra_column_name, check_exists=False)
        )
        # Now we can concatenate
        question_doc_sections_dp = mk.concat(question_doc_dp["section_dp"])

    # Return all the DataPanels we have so far
    return SimpleNamespace(
        doc_dp=doc_dp,
        story_doc_dp=story_doc_dp,
        question_doc_dp=question_doc_dp,
        story_doc_sections_dp=story_doc_sections_dp,
        question_doc_sections_dp=question_doc_sections_dp,
    )


doc_data = load_fairytale_qa_doc()
story_doc_dp = doc_data.story_doc_dp
story_doc_sections_dp = doc_data.story_doc_sections_dp

##### Build the interface #####

# Start the Meerkat GUI server
gui_info = mk.gui.start(shareable=False)

# Wrap the DataPanel with a `Pivot`
# This makes it a starting point for the interface
story_doc_dp = mk.gui.Pivot(story_doc_dp)
story_doc_sections_dp = mk.gui.Pivot(story_doc_sections_dp)

# Make a Choice component: this allows us to choose a document
doc_id_choice = mk.gui.Choice(
    value=story_doc_dp["id"][0],  # the default value is the first document id
    choices=list(story_doc_dp["id"]),  # choices are the document ids
)
# ...access the chosen document id with `choice.value`
# this is a `Store` object whose value can be accessed with `<store>.value`
# Stores are a way to keep track of state that is changing as the user interacts
# with the frontend. Their value will automatically be synced by Meerkat
# with the frontend.

# Hacky way to get the index of the chosen document id
# We're adding support for primary keys in Meerkat, so this will be
# cleaned up in a couple of weeks
_index = dict(zip(story_doc_dp["id"], range(len(story_doc_dp["id"]))))


@mk.gui.interface_op
def choose_doc(doc_id):
    """
    This is an interactive operation (decorate with `@mk.gui.interface_op`)
    that is retriggered whenever the user changes the values of the arguments.

    Here, when the user chooses something in the `Choice` component
    i.e. changes the value of `doc_id_choice.value`, this function will
    - get the index of the chosen `doc_id`
    - return the corresponding row from the `story_doc_dp` DataPanel,
    for column `section_dp`

    Important: this code needs to be inside a function, e.g. if you
    copy and paste this code outside of the function, it won't work.
    This is because the `@mk.gui.interface_op` decorator does some
    work to unwrap the `doc_id` Store.
    """
    # Get the document index
    doc_idx = _index[doc_id]
    return story_doc_dp["section_dp"][doc_idx]


# Create the Document component
document = mk.gui.Document(
    dp=choose_doc(doc_id_choice.value),  # sections of the chosen document
    doc_column="text",  # the column that contains document text
)

# Launch the interface
mk.gui.Interface(
    components=[doc_id_choice, document],
).launch()
