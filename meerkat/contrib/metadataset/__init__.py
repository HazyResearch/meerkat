import glob
import json
from copy import deepcopy
from functools import partial

import meerkat as mk


def simplify_scenegraph(scenegraph):
    """
    Extract only the `objects` key in the scenegraph, and process it to resolve all
    objects ids to their names.

    Args:
        scenegraph: a scenegraph dictionary from Visual Genome

    Returns:
        dictionary containing objects inside the scenegraph
    """
    object_to_name_mapping = {}
    for obj_id, obj_info in scenegraph['objects'].items():
        object_to_name_mapping[obj_id] = obj_info['name']

    objects = {}
    for obj_id, obj_info in scenegraph['objects'].items():
        objects[obj_info['name']] = deepcopy(obj_info)
        objects[obj_info['name']]['relations'] = []
        for relation in obj_info['relations']:
            objects[obj_info['name']]['relations'].append(
                (relation['name'], object_to_name_mapping[relation['object']]))

    return objects


def fetch_category_info(x):
    for obj_id, obj_info in x['scenegraph']['objects'].items():
        if obj_info['name'] == x['category']:
            return obj_info
    return {}


def fetch_scenegraph(scenegraphs, x):
    try:
        return scenegraphs[x]
    except KeyError:
        return {}


def load_metadataset(
        path: str = '/home/common/datasets/metadataset/',
):
    # DataPanel constructed from all images contained in `path`
    all_images = []
    labels = []
    for path in glob.glob(f'{path}/proc/*'):
        images = list(glob.glob(f'{path}/proc/*/*'))
        all_images.extend(images)
        # label information comes from the path
        labels.extend([path.split("/")[-1]] * len(images))

    # one category per image
    categories = [
        path.split("/")[-2].split("(")[-1].replace(")", "")
        for path in all_images
    ]

    dp = mk.DataPanel({
        'image': mk.ImageColumn.from_filepaths(
            filepaths=all_images,
        ),
        'label': labels,
        'category': categories,
    })

    # ID column extracted from the image paths
    dp['id'] = [im.data.split("/")[-1].strip(".jpg") for im in dp.lz['image'].lz]

    train_scenegraphs = json.load(
        open(f'{path}/train_sceneGraphs.json')
    )
    dp['scenegraph'] = mk.LambdaColumn(dp['id'],
                                       partial(fetch_scenegraph, train_scenegraphs))
    dp['objects'] = mk.LambdaColumn(dp['scenegraph'], simplify_scenegraph)

    return dp
