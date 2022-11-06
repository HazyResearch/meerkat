# flake8: noqa
import argparse
from glob import glob
from types import SimpleNamespace

import dtw
import numpy as np
from thefuzz import fuzz

import meerkat as mk

failed_indices = set()


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


def load_fairytale_qa_doc(doc_path):
    # Create a Document DataFrame
    doc_df = mk.DataFrame(
        {"path": glob(doc_path + "/*/*.csv")}
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


def load_sentence_csv(row):
    """
    Loader for a single sentence csv file. This function takes a row of the sentence_df DataFrame
    and returns a DataFrame with the contents of the csv file.

    row: a row from the sentence_df DataFrame
    """
    # Load the csv file from disk as a DataFrame
    sentence_data = mk.DataFrame.from_csv(row["path"])  # access the 'path'
    # Just add a sentence_id column, so we know which document this CSV came from
    sentence_data["sentence_id"] = [row["id"]] * len(sentence_data)  # access the 'id'
    # Drop the `document_id` column, because we already have the `id` column
    sentence_data = sentence_data.drop("document_id")
    return sentence_data


def load_fairytale_qa_sentence(sentence_path):
    # Create a Sentence DataFrame
    sentence_df = mk.DataFrame(
        {"path": glob(sentence_path + "/*/*.csv")}
    )  # put in a column called 'path' with all the paths to the csv files

    # Create some useful columns
    sentence_df["source"] = sentence_df["path"].apply(
        lambda x: x.split("/")[-2]
    )  # the source of the document e.g. "anderson-fairybook"
    sentence_df["name"] = sentence_df["path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )  # the name of the document e.g. "brave-tin-soldier-story.csv"
    sentence_df["id"] = (
        sentence_df["source"] + "/" + sentence_df["name"]
    )  # the id of the document e.g. "anderson-fairybook/brave-tin-soldier-story"

    sentence_df["type"] = (
        sentence_df["name"].str.split("-").str.get(-1)
    )  # the type of document among [story, mrs] -- here we use pandas string methods directly on the column
    sentence_df["sentence_df"] = sentence_df.map(
        load_sentence_csv
    )  # .map will run the function on each line of the DataFrame and put the result in a new column called 'sentence_df'

    # Split the sentence_df DataFrame into two DataFrames, one for stories and one for mrs
    story_sentence_df = sentence_df[sentence_df["type"] == "story"]
    mrs_sentence_df = sentence_df[sentence_df["type"] == "mrs"]
    story_sentence_df.shape, mrs_sentence_df.shape

    # Like before, we can now access all the DataFrames inside story_sentence_df, concatenate them and get out a single DataFrame
    story_sentence_sections_df = mk.concat(
        story_sentence_df["sentence_df"].filter(lambda x: len(x) > 0)
    )  # need to filter out the empty DataFrames
    mrs_sentence_sections_df = mk.concat(
        mrs_sentence_df["sentence_df"].filter(lambda x: len(x) > 0)
    )  # need to filter out the empty DataFrames
    sentence_sections_df = mk.concat(
        sentence_df["sentence_df"].filter(lambda x: len(x) > 0)
    )  # need to filter out the empty DataFrames

    # Return all the DataFrames we have so far
    return SimpleNamespace(
        sentence_df=sentence_df,
        story_sentence_df=story_sentence_df,
        mrs_sentence_df=mrs_sentence_df,
        story_sentence_sections_df=story_sentence_sections_df,
        mrs_sentence_sections_df=mrs_sentence_sections_df,
        sentence_sections_df=sentence_sections_df,
    )


def _sentence_in_section_df(sentence, section_df):
    """
    Take a dataframe with sections and a sentence, and
    return a column containing a fuzzy match score to say if the
    sentence is in the corresponding section.
    """
    return section_df["text"].map(lambda x: fuzz.partial_ratio(sentence, x))


def _sentence_df_in_section_df(sentence_df, section_df, index):
    """
    Take a dataframe with sections and a dataframe with sentences,
    and return a column. Each row of the column contains the section
    that the sentence is most similar to (and likely contained in).
    """

    try:
        cost_matrix = -sentence_df["text"].map(
            lambda x: _sentence_in_section_df(x, section_df)
        )
        # Use dynamic time warping to find the best alignment (this ensures no jumps)
        # Note: this simple version below (using section with argmax similarity) doesn't work!
        # correspondence = sentence_df['text'].map(lambda x: _sentence_in_section_df(x, section_df).argmax())
        correspondence = dtw.dtw(
            cost_matrix.astype(np.float64),
            window_type="none",
            step_pattern="asymmetric",
        ).index2
        sentence_df["section"] = correspondence
        return correspondence
    except:
        failed_indices.add(index)
        return []


def create_story_sentence_df(data_dir):
    # Set up paths
    doc_path = data_dir + "FairytaleQA_Dataset/split_by_origin"
    sentence_path = data_dir + "FairytaleQA_Dataset_Sentence_Split/split_by_origin"

    # Load the data
    doc_data = load_fairytale_qa_doc(doc_path)
    sentence_data = load_fairytale_qa_sentence(sentence_path)
    story_doc_df = doc_data.story_doc_df
    story_sentence_df = sentence_data.story_sentence_df

    story_df = mk.merge(story_sentence_df, story_doc_df, on="id")[
        "id", "sentence_df", "section_df"
    ]

    """
    The sentence <-> section correspondence is missing from the data, so we need to add it.
    pip install dtw-python thefuzz[speedup]
    in the conda env
    """

    # Takes around 3 minutes to run
    story_df.map(
        lambda x, i: list(
            _sentence_df_in_section_df(x["sentence_df"], x["section_df"], i)
        ),
        pbar=True,
        with_indices=True,
    )
    print("Failed indices:", failed_indices)

    # Manually fix the two stories that failed: {58, 102}
    story_df["sentence_df"][58] = mk.concat(
        story_df[58]["section_df"].map(
            lambda x: mk.DataPanel(
                [
                    {
                        "text": (s + "." if not s.endswith(".") else s),
                        "section": x["section"],
                        "sentence_id": x["doc_id"],
                    }
                    for s in x["text"].split(". ")
                ]
            )
        )
    )
    story_df["sentence_df"][102] = mk.concat(
        story_df[102]["section_df"].map(
            lambda x, i: mk.DataPanel(
                [
                    {
                        "text": (s + "." if not s.endswith(".") else s),
                        "section": i + 1,
                        "sentence_id": x["doc_id"],
                    }
                    for s in x["section"].split(". ")
                ]
            ),
            with_indices=True,
        )
    )

    # Write the data to disk
    mk.concat(story_df["sentence_df"]).write("story_sentence_processed_df.mk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/krandiash/Desktop/workspace/projects/datasci/data/FairytaleQA_Dataset/",
    )
    args = parser.parse_args()

    # Setup the data
    create_story_sentence_df(args.data_dir)
