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
    default="/Users/dakuowang/Desktop/FairytaleQA_Dataset/",
)
args = parser.parse_args()


##### Load the data #####

# Set up your paths
DATA_DIR = args.data_dir
DOC_PATH = DATA_DIR + "FairytaleQA_Dataset/split_by_origin"
SENTENCE_PATH = DATA_DIR + "FairytaleQA_Dataset_Sentence_Split/split_by_origin"


def load_doc_csv(row):
    """
    Loader for a single document csv file. This function takes a row of the doc_df DataFrame
    and returns a DataFrame with the contents of the csv file.

    row: a row from the doc_df DataFrame
    """
    # Load the csv file from disk as a DataFrame
    doc_data = mk.DataFrame.from_csv(row["path"])  # access the 'path'
    # Just add a doc_id column, so we know which document this CSV came from
    doc_data["doc_id"] = [row["id"]] * len(doc_data)  # access the 'id'
    return doc_data


def load_fairytale_qa_doc():
    # Create a Document DataFrame
    doc_df = mk.DataFrame(
        {"path": glob(DOC_PATH + "/*/*.csv")}
    )  # put in a column called 'path' with all the paths to the csv files

    # Create some useful columns
    doc_df["source"] = doc_df["path"].apply(
        lambda x: x.split("/")[-2]
    )  # the source of the document e.g. "anderson-fairybook"
    doc_df["name"] = doc_df["path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )  # the name of the document e.g. "brave-tin-soldier-story.csv"
    doc_df["id"] = (
        doc_df["source"] + "/" + doc_df["name"]
    )  # the id of the document e.g. "anderson-fairybook/brave-tin-soldier-story"

    doc_df["type"] = (
        doc_df["name"].str.split("-").str.get(-1)
    )  # the type of document among [story, questions] -- here we use pandas string methods directly on the column

    doc_df["section_df"] = doc_df.map(
        load_doc_csv
    )  # .map will run the function on each line of the DataFrame and put the result in a new column called 'section_df'

    # Split the doc_df DataFrame into two DataFrames, one for stories and one for questions
    story_doc_df = doc_df[doc_df["type"] == "story"]
    question_doc_df = doc_df[doc_df["type"] == "questions"]

    # Magic: we can now access all the DataFrames inside story_doc_df, concatenate them and get out a single DataFrame
    story_doc_sections_df = mk.concat(story_doc_df["section_df"])

    # Similarly, we can do this for the question DataFrames
    try:
        # This doesn't work, because a couple of the question DataFrames have different columns
        question_doc_sections_df = mk.concat(question_doc_df["section_df"])
    except:
        # We can check which rows have different columns
        has_extra_column = np.where(
            question_doc_df.map(lambda x: len(x["section_df"].columns)) == 16
        )[0]
        print(
            "These rows have an extra column, while all the others have 15 columns",
            has_extra_column,
        )  # 47, 109
        extra_column_name = set(question_doc_df[47]["section_df"].columns) - set(
            question_doc_df[0]["section_df"].columns
        )
        print("The extra column is called", extra_column_name)  # called "comments"
        # We can fix this by dropping the extra column if it exists
        question_doc_df["section_df"] = question_doc_df["section_df"].map(
            lambda x: x.drop(extra_column_name, check_exists=False)
        )
        # Now we can concatenate
        question_doc_sections_df = mk.concat(question_doc_df["section_df"])

    # Return all the DataFrames we have so far
    return SimpleNamespace(
        doc_df=doc_df,
        story_doc_df=story_doc_df,
        question_doc_df=question_doc_df,
        story_doc_sections_df=story_doc_sections_df,
        question_doc_sections_df=question_doc_sections_df,
    )


doc_data = load_fairytale_qa_doc()
story_doc_df = doc_data.story_doc_df
story_doc_sections_df = doc_data.story_doc_sections_df

##### Build the interface #####

# Start the Meerkat GUI server
gui_info = mk.gui.start(shareable=False)

# Wrap the DataFrame with a `Reference`
# This makes it a starting point for the interface
story_doc_df = mk.gui.Reference(story_doc_df)
story_doc_sections_df = mk.gui.Reference(story_doc_sections_df)

# Make a Choice component: this allows us to choose a document
doc_id_choice = mk.gui.Choice(
    value=story_doc_df["id"][0],  # the default value is the first document id
    choices=list(story_doc_df["id"]),  # choices are the document ids
)
# ...access the chosen document id with `choice.value`
# this is a Store object whose value can be accessed with `<store>.value`

# Hacky way to get the index of the chosen document id
# We're adding support for primary keys in Meerkat, so this will be
# cleaned up in a couple of weeks
_index = dict(zip(story_doc_df["id"], range(len(story_doc_df["id"]))))


@mk.gui.interface_op
def choose_doc(doc_id):
    """
    This is an interactive operation (decorate with `@mk.gui.interface_op`)
    that is retriggered whenever the user changes the values of the arguments.

    Here, when the user chooses something in the `Choice` component
    i.e. changes the value of `doc_id_choice.value`, this function will
    - get the index of the chosen `doc_id`
    - return the corresponding row from the `story_doc_df` DataFrame,
    for column `section_df`

    Important: this code needs to be inside a function
    """
    # Get the document index
    doc_idx = _index[doc_id]
    return story_doc_df["section_df"][doc_idx]


# Create the Document component
document = mk.gui.Document(
    df=choose_doc(doc_id_choice.value),  # sections of the chosen document
    doc_column="text",  # the column that contains document text
)

# Launch the interface
mk.gui.Interface(
    components=[doc_id_choice, document],
).launch()
