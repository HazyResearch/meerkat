"""A demo for segmenting images with the Segment Anything Model (SAM).

This demo requires access to the Segment Anything Model (SAM) model.
https://github.com/facebookresearch/segment-anything
"""
# flake8: noqa: E402

import os
from typing import List

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image as PILImage
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

import meerkat as mk
from meerkat.interactive.formatter.image import ImageFormatterGroup

# Initialize SAM model
model_type = "vit_b"
sam_checkpoint = os.path.expanduser(
    "~/.cache/segment_anything/models/sam_vit_b_01ec64.pth"
)
device = "mps"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


@mk.reactive()
def prepare_image(image):
    """Prepare the image for the SAM model."""
    if isinstance(image, PIL.Image.Image):
        image = np.asarray(image)
        # Grayscale -> RGB.
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.concatenate([image, image, image], axis=-1)
    return np.asarray(image)


@mk.reactive()
def predict(image, boxes: mk.Store[List], segmentations: mk.Store[List]):
    """Segment the image with the SAM model.

    When ``boxes`` changes, this function will be called to segment the image.
    The segmentation will be appended to the segmentations list.
    This is a bit hacky, but it should work.

    Note:
        Because this will run any time ``boxes`` changes, it will also run
        when boxes are deleted, which can lead to unexpected behavior.

    Returns:
        np.ndarray: The colorized segmentation mask.
    """
    image = prepare_image(image)

    predictor.set_image(np.asarray(image))

    combined_segmentations = []
    if len(boxes) > 0:
        # Assume the last box is the most recently added box.
        box = boxes[-1]
        box_array = np.asarray(
            [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]]
        )
        masks, scores, _ = predictor.predict(box=box_array)
        best_idx = np.argsort(scores)[-1]
        predictions = [(masks[best_idx], box["category"])]

        segmentations.extend(predictions)
        combined_segmentations.extend(segmentations)

    # Do this to trigger the reactive function.
    # segmentations.set(segmentations)
    return mk.Store(combined_segmentations, backend_only=True)


@mk.reactive()
def predict_point(
    image, points: mk.Store[List], segmentations: mk.Store[List], selected_category: str
):
    """Segment the image with the SAM model.

    When ``boxes`` changes, this function will be called to segment the image.
    The segmentation will be appended to the segmentations list.
    This is a bit hacky, but it should work.

    Note:
        Because this will run any time ``boxes`` changes, it will also run
        when boxes are deleted, which can lead to unexpected behavior.

    Returns:
        np.ndarray: The colorized segmentation mask.
    """
    image = prepare_image(image)

    predictor.set_image(np.asarray(image))

    combined_segmentations = []
    if len(points) > 0:
        points = np.asarray([[point["x"], point["y"]] for point in points])
        # Assume that all points are foreground points.
        point_labels = np.ones(len(points))
        masks, scores, _ = predictor.predict(
            point_coords=points, point_labels=point_labels
        )
        best_idx = np.argsort(scores)[-1]

        # If there is already a prediction for this category,
        # we assume that we are operating on this prediction.
        # Remove this mask and add the new one.
        # TODO: Use this mask as a prompt for the prediction.
        categories = [seg[1] for seg in segmentations]
        if selected_category in categories:
            segmentations[categories.index(selected_category)] = (
                masks[best_idx],
                selected_category,
            )
        else:
            predictions = [(masks[best_idx], selected_category)]
            segmentations.extend(predictions)
    combined_segmentations.extend(segmentations)

    # Do this to trigger the reactive function.
    # segmentations.set(segmentations)
    return mk.Store(combined_segmentations, backend_only=True)


@mk.reactive()
def clear_points(points: List, selected_category: str):
    """When the selected category changes, clear the points."""
    # We do not want to trigger segmentation to re-run here.
    # So we use a hacky solution and clear the points.
    points.clear()
    return points


@mk.reactive()
def get_categories(segmentations: List[str]):
    return [seg[1] for seg in segmentations]


@mk.reactive()
def get_img_and_annotations(idx: int):
    row = images[idx]

    img = row["image"]()
    segmentations = row["segmentation"]
    boxes = row["boxes"]
    points = row["points"]

    # Pushing up large arrays to the frontend is slow.
    # Only maintain the image and segmentations on the backend.
    # The serialized version of the image and segmentations will be sent to the frontend.
    img = mk.Store(img, backend_only=True)
    segmentations = mk.Store(segmentations, backend_only=True)
    boxes = mk.Store(boxes, backend_only=False)
    points = mk.Store(points, backend_only=False)

    return mk.Store((img, segmentations, boxes, points), backend_only=True)


@mk.endpoint()
def increment(idx: mk.Store[int], annotator: mk.gui.ImageAnnotator):
    idx.set(idx + 1)  # set max guard
    annotator.clear_annotations()
    return idx


@mk.endpoint()
def decrement(idx: mk.Store[int], annotator: mk.gui.ImageAnnotator):
    idx.set(max(idx - 1, 0))
    annotator.clear_annotations()
    return idx


# Hacky way to get the annotations written to the dataframe
# with reactive statements.
@mk.reactive()
def update_annotations(idx: int, ann_type: str, annotations: List[str]):
    images[ann_type][idx] = annotations


# Build the dataframe.
files = [
    # "https://kulapartners.com/wp-content/uploads/2017/06/multiple-personas-hero.jpg",
    "https://3acf3052-cdn.agilitycms.cloud/images/service/KNEE%20SAGITTAL.jpg",
    # "https://www.mercurynews.com/wp-content/uploads/2022/01/BNG-L-WARRIORS-0122-28.jpg?w=1024"
]
images = mk.DataFrame({"image": mk.files(files, type="image")})
# Add an empty list for each of the annotations.
# This is a simple way of managing to keep track of the annotations in the dataframe.
images["segmentation"] = [[] for _ in range(len(images))]
images["boxes"] = [[] for _ in range(len(images))]
images["points"] = [[] for _ in range(len(images))]


idx = mk.Store(0, backend_only=True)
selected_category = mk.Store("")
img, segmentations, boxes, points = get_img_and_annotations(idx)
update_annotations(idx, "boxes", boxes)
update_annotations(idx, "points", points)
points = clear_points(points=points.unmark(), selected_category=selected_category)

box_segmentations = predict(
    image=img,
    boxes=boxes,
    segmentations=segmentations,
)
point_segmentations = predict_point(
    image=img,
    points=points,
    segmentations=segmentations,
    selected_category=selected_category.unmark(),
)

with mk.magic():
    combined_segmentations = box_segmentations + point_segmentations

# Image annotator.
selected_category.mark()
categories = get_categories(segmentations)
annotator = mk.gui.ImageAnnotator(
    img,
    categories=[],  # categories["name"].tolist(),
    segmentations=combined_segmentations,
    points=points,
    boxes=boxes,
    selected_category=selected_category,
)

# Layout.
component = mk.gui.html.gridcols3(
    [
        mk.gui.Button(
            title="",
            icon="ChevronLeft",
            on_click=decrement.partial(idx=idx, annotator=annotator),
        ),
        annotator,
        mk.gui.Button(
            title="",
            icon="ChevronRight",
            on_click=increment.partial(idx=idx, annotator=annotator),
        ),
    ],
    classes="h-screen grid-cols-[auto_6fr_auto]",
)

page = mk.gui.Page(component, id="SAM")
page.launch()
