import json
import os
from typing import Dict, List, Union

import meerkat as mk


def build_audioset_df(
    dataset_dir: str,
    splits: List[str] = None,
    audio_column: bool = True,
    overwrite: bool = False,
) -> Dict[str, mk.DataFrame]:
    """Build DataFrames for the audioset dataset downloaded to ``dataset_dir``.
    By default, the resulting DataFrames will be written to ``dataset_dir``
    under the filenames "audioset_examples.mk" and "audioset_labels.mk". If
    these files already exist and ``overwrite`` is False, the DataFrames will
    not be built anew, and instead will be simply loaded from disk.

    Args:
        dataset_dir: The directory where the dataset is stored
        download: Whether to download the dataset
        splits: A list of splits to include. Defaults to ["eval_segments"].
            Other splits: "balanced_train_segments", "unbalanced_train_segments".
        audio_column (bool): Whether to include a :class:`~meerkat.AudioColumn`.
            Defaults to True.
        overwrite (bool): Whether to overwrite existing DataFrames saved to disk.
            Defaults to False.
    """

    if splits is None:
        splits = ["eval_segments"]

    if (
        os.path.exists(os.path.join(dataset_dir, "audioset_examples.mk"))
        and os.path.exists(os.path.join(dataset_dir, "audioset_labels.mk"))
        and not overwrite
    ):
        return {
            "examples": mk.DataFrame.read(
                os.path.join(dataset_dir, "audioset_examples.mk")
            ),
            "labels": mk.DataFrame.read(
                os.path.join(dataset_dir, "audioset_labels.mk")
            ),
        }

    dfs = []
    label_rows = []
    for split in splits:
        if not os.path.exists(os.path.join(dataset_dir, f"{split}.csv")):
            raise ValueError(f"{split}.csv not found.")

        df = mk.DataFrame.from_csv(
            os.path.join(dataset_dir, f"{split}.csv"),
            names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
            skiprows=3,
            delimiter=", ",
            engine="python",  # suppresses warning
        )

        df["split"] = [split for i in range(len(df))]
        df["audio_path"] = df.map(
            lambda row: os.path.join(
                dataset_dir,
                split,
                "YTID={}_st={}_et={:.0f}.wav".format(
                    row["YTID"], row["start_seconds"], row["end_seconds"]
                ),
            ),
        )

        label_rows.extend(
            [
                {"YTID": row["YTID"], "label_id": label_id}
                for row in df[["positive_labels", "YTID"]]
                for label_id in row["positive_labels"].strip('"').split(",")
            ]
        )
        df.remove_column("positive_labels")

        # Filter missing audio
        df = df[df["audio_path"].apply(os.path.exists)]

        if audio_column:
            df["audio"] = mk.AudioColumn(df["audio_path"])
        dfs.append(df)

    dataset = {
        "examples": mk.concat(dfs) if len(dfs) > 1 else dfs[0],
        "labels": mk.DataFrame(label_rows),
    }

    dataset["examples"].write(os.path.join(dataset_dir, "audioset_examples.mk"))
    dataset["labels"].write(os.path.join(dataset_dir, "audioset_labels.mk"))

    return dataset


def build_ontology_df(dataset_dir: str) -> Dict[str, mk.DataFrame]:
    """Build a DataFrame from the ontology.json file.

    Args:
        dataset_dir: The directory where the ontology.json file is stored
    """
    data = json.load(open(os.path.join(dataset_dir, "ontology.json")))
    df = mk.DataFrame.from_dict(data)
    relations = [
        {"parent_id": row["id"], "child_id": child_id}
        for row in df[["id", "child_ids"]]
        for child_id in row["child_ids"]
    ]
    df.remove_column("child_ids")
    df.remove_column("positive_examples")
    df.remove_column("restrictions")

    return {"sounds": df, "relations": mk.DataFrame(relations)}


def find_submids(
    id: Union[List[str], str],
    relations: mk.DataFrame = None,
    dataset_dir: str = None,
) -> List[str]:
    """Returns a list of IDs of all subcategories of an audio category.

    Args:
        ids: ID or list of IDs for which to find the subcategories
        df: A DataFrame built from the ontology.json file.
        dataset_dir: Alternatively, the directory where the ontology.json file is stored
            can be provided to construct a DataFrame
    """

    if (not relations) == (not dataset_dir):
        raise ValueError("Must pass either `relations` or `dataset_dir` but not both.")

    if dataset_dir is not None:
        ontology = build_ontology_df(dataset_dir=dataset_dir)
        relations = ontology["relations"]

    submids = set()
    queue = id if isinstance(id, list) else [id]
    while len(queue):
        parent_mid = queue[0]
        queue.pop(0)
        child_ids = relations[relations["parent_id"] == parent_mid]["child_id"]
        queue.extend(child_ids)
        submids.update(child_ids)

    return list(submids)
