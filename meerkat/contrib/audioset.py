import json
import os
from typing import List, Union

import meerkat as mk


def build_audioset_dp(
    dataset_dir: str,
    splits: List[str] = ["eval_segments"],
    batch_size=32,
    num_workers: int = 8,
    pbar: bool = True,
    audio_column: bool = True,
):
    """
    Build a DataPanel from the audioset dataset

    Args:
        dataset_dir: The directory where the dataset is stored
        download: Whether to download the dataset
        splits: A list of splits to include. Defaults to eval_segments.
            Other values: "balanced_train_segments", "unbalanced_train_segments".
    """

    dps = []
    for split in splits:
        if not os.path.exists(os.path.join(dataset_dir, f"{split}.csv")):
            raise ValueError(f"{split}.csv not found.")

        dp = mk.DataPanel.from_csv(
            os.path.join(dataset_dir, f"{split}.csv"),
            names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
            skiprows=3,
            delimiter=", ",
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
            pbar=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Filter missing audio
        dp = dp.filter(
            lambda x: True if os.path.exists(x["audio_path"]) else False,
            pbar=pbar,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        # TODO(Priya): Filter out empty files
        dp["positive_labels"] = dp["positive_labels"].map(
            lambda labels: labels.split(","),
            pbar=pbar,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if audio_column:
            dp["audio"] = mk.AudioColumn(dp["audio_path"])

        dps.append(dp)

    return mk.concat(dps) if len(dps) > 1 else dps[0]


def build_ontology_dp(dataset_dir: str):
    """
    Build a DataPanel from the ontology.json file

    Args:
        dataset_dir: The directory where the ontology.json file is stored
    """
    data = json.load(open(os.path.join(dataset_dir, "ontology.json")))
    dp = mk.DataPanel.from_dict(data)
    return dp


def find_submids(
    id: Union[List[str], str], dp: mk.DataPanel = None, dataset_dir: str = None
):
    """
    Returns a list of IDs of all subcategories of an audio category

    Args:
        ids: ID or list of IDs for which to find the subcategories
        dp: A DataPanel built from the ontology.json file.
        dataset_dir: Alternatively, the directory where the ontology.json file is stored
            can be provided to construct a DataPanel
    """

    if not dp and not dataset_dir:
        raise ValueError("One of ontology DataPanel or directory path is required.")

    if not dp:
        dp = build_ontology_dp(dataset_dir)

    submids = set()
    queue = id if isinstance(id, list) else [id]
    while len(queue):
        parent_mid = queue[0]
        queue.pop(0)
        child_ids = dp[dp["id"] == parent_mid]["child_ids"].data[0]
        queue.extend(child_ids)
        submids.update(child_ids)

    return list(submids)
