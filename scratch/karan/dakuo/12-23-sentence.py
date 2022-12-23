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

# Load functions from the script '12-23-clean-data.py'
utils = importlib.import_module("12-23-clean-data")

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

##### Build the interface #####

# Start the Meerkat GUI server
gui_info = mk.gui.start(shareable=False)

search_keywords = mk.gui.Textbox(title="type keywords here")

story_sentence_df = story_sentence_df.head(200)
story_sentence_df = mk.embed(story_sentence_df, input="text", encoder="clip", out_col="embed")

@mk.gui.reactive
def search_sentences_by_keywords(sentence_df: mk.DataFrame, keywords: str = None):
    kw_df = mk.DataFrame({"text": [keywords or "abc"]})
    kw_embed = mk.embed(kw_df, input="text", encoder="clip", out_col="embed")
    sentence_df['scores'] = sentence_df['embed'] @ kw_embed['embed'][0].T
    sort_by_keyword_df = sentence_df.sort(by="scores", ascending=False)
    sort_by_keyword_df.create_primary_key("id")
    sort_by_keyword_df["section"] = sort_by_keyword_df["id"]
    return sort_by_keyword_df.head(10)


with mk.gui.react():
    document_df = search_sentences_by_keywords(
        story_sentence_df, search_keywords.text
    )

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

# Launch the interface
mk.gui.Interface(
    component=RowLayout(components=[search_keywords, document]),
).launch()
