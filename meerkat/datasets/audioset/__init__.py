import json
import os
from typing import Dict, List, Union

import meerkat as mk


def build_audioset_dp(
    dataset_dir: str,
    splits: List[str] = None,
    audio_column: bool = True,
    overwrite: bool = False,
) -> Dict[str, mk.DataPanel]:
    """Build DataPanels for the audioset dataset downloaded to ``dataset_dir``.
    By default, the resulting DataPanels will be written to ``dataset_dir``
    under the filenames "audioset_examples.mk" and "audioset_labels.mk". If
    these files already exist and ``overwrite`` is False, the DataPanels will
    not be built anew, and instead will be simply loaded from disk.

    Args:
        dataset_dir: The directory where the dataset is stored
        download: Whether to download the dataset
        splits: A list of splits to include. Defaults to ["eval_segments"].
            Other splits: "balanced_train_segments", "unbalanced_train_segments".
        audio_column (bool): Whether to include a :class:`~meerkat.AudioColumn`.
            Defaults to True.
        overwrite (bool): Whether to overwrite existing DataPanels saved to disk.
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
            "examples": mk.DataPanel.read(
                os.path.join(dataset_dir, "audioset_examples.mk")
            ),
            "labels": mk.DataPanel.read(
                os.path.join(dataset_dir, "audioset_labels.mk")
            ),
        }

    dps = []
    label_rows = []
    for split in splits:
        if not os.path.exists(os.path.join(dataset_dir, f"{split}.csv")):
            raise ValueError(f"{split}.csv not found.")

        dp = mk.DataPanel.from_csv(
            os.path.join(dataset_dir, f"{split}.csv"),
            names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
            skiprows=3,
            delimiter=", ",
            engine="python",  # suppresses warning
        )

        dp["split"] = [split for i in range(len(dp))]
        dp["audio_path"] = dp.map(
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
                for row in dp[["positive_labels", "YTID"]]
                for label_id in row["positive_labels"].strip('"').split(",")
            ]
        )
        dp.remove_column("positive_labels")

        # Filter missing audio
        dp = dp.lz[dp["audio_path"].apply(os.path.exists)]

        if audio_column:
            dp["audio"] = mk.AudioColumn(dp["audio_path"])
        dps.append(dp)

    dataset = {
        "examples": mk.concat(dps) if len(dps) > 1 else dps[0],
        "labels": mk.DataPanel(label_rows),
    }

    dataset["examples"].write(os.path.join(dataset_dir, "audioset_examples.mk"))
    dataset["labels"].write(os.path.join(dataset_dir, "audioset_labels.mk"))

    return dataset


def build_ontology_dp(dataset_dir: str) -> Dict[str, mk.DataPanel]:
    """Build a DataPanel from the ontology.json file.

    Args:
        dataset_dir: The directory where the ontology.json file is stored
    """
    data = json.load(open(os.path.join(dataset_dir, "ontology.json")))
    dp = mk.DataPanel.from_dict(data)
    relations = [
        {"parent_id": row["id"], "child_id": child_id}
        for row in dp[["id", "child_ids"]]
        for child_id in row["child_ids"]
    ]
    dp.remove_column("child_ids")
    dp.remove_column("positive_examples")
    dp.remove_column("restrictions")

    return {"sounds": dp, "relations": mk.DataPanel(relations)}


def find_submids(
    id: Union[List[str], str],
    relations: mk.DataPanel = None,
    dataset_dir: str = None,
) -> List[str]:
    """Returns a list of IDs of all subcategories of an audio category.

    Args:
        ids: ID or list of IDs for which to find the subcategories
        dp: A DataPanel built from the ontology.json file.
        dataset_dir: Alternatively, the directory where the ontology.json file is stored
            can be provided to construct a DataPanel
    """

    if (not relations) == (not dataset_dir):
        raise ValueError("Must pass either `relations` or `dataset_dir` but not both.")

    if dataset_dir is not None:
        ontology = build_ontology_dp(dataset_dir=dataset_dir)
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
