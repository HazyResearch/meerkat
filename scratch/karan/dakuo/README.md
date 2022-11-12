# Scripts

Run the following script to generate the dataframe:
```bash
conda activate meerkat
pip install dtw-python thefuzz[speedup]
python 11-04-clean-data.py --data_dir [/path/to/FairytaleQA_Dataset]
```
This script tries to align the paragraph-sentence pairs in a hacky way, to find the `section`
associated with each sentence (this information seems to be missing in the data). See script for details.

You should modify this script to make sure that the alignment is correct for all stories
(currently it is not).
(I recommend running the Spacy pipeline on the paragraphs to regenerate the sentence boundaries.)

After this runs (takes around 3 minutes), run the following script to spin up the labeling interface:
```bash
python 11-04-sentence.py --data_dir [/path/to/FairytaleQA_Dataset]
```
Whatever you label will be saved in the `labels.jsonl` file in the same directory!
