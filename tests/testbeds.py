"""A collection of simple testbeds to build test cases."""
from copy import deepcopy

from mosaic.datapanel import DataPanel
from mosaic.tools.identifier import Identifier


class MockTestBedv0:
    """Simple mock dataset with 6 examples."""

    def __init__(self):
        # Create a fake batch of data
        self.batch = {
            "text": [
                "The man is walking.",
                "The man is running.",
                "The woman is sprinting.",
                "The woman is resting.",
                "The hobbit is flying.",
                "The hobbit is swimming.",
            ],
            "label": [0, 0, 1, 1, 0, 0],
            "z": [1, 0, 1, 0, 1, 0],
            "fast": [False, True, True, False, False, False],
            "metadata": [
                {"source": "real"},
                {"source": "real"},
                {"source": "real"},
                {"source": "real"},
                {"source": "fictional"},
                {"source": "fictional"},
            ],
        }
        # Create a fake dataset
        self.dataset = DataPanel.from_batch(
            self.batch,
            identifier=Identifier(_name="MockDataPane", version="1.0"),
        )

        # Keep a copy of the original
        self.original_dataset = deepcopy(self.dataset)

        assert len(self.dataset) == 6

    def test_attributes(self):
        # Both datasets use the same cache files for backing
        print(self.dataset.cache_files)
        print(self.original_dataset.cache_files)
        print(self.dataset.identifier)

    def problems(self):
        # FIXME(karan): this shouldn't be happening: why is otherlabel disappearing here
        with self.assertRaises(AssertionError):
            # Create an additional integer column in the dataset
            dataset = self.testbed.dataset.map(lambda x: {"otherlabel": x["label"] + 1})
            dataset_0_0 = self.cachedop(dataset, columns=["label"])
            self.assertTrue("otherlabel" in dataset_0_0.column_names)


class MockTestBedv1:
    """Simple mock dataset with 4 examples containing pairs of sentences."""

    def __init__(self):
        # Create a fake dataset
        self.dataset = DataPanel.from_batch(
            {
                "text_a": [
                    "Before the actor slept, the senator ran.",
                    "The lawyer knew that the judges shouted.",
                    "If the actor slept, the judge saw the artist.",
                    "The lawyers resigned, or the artist slept.",
                ],
                "text_b": [
                    "The actor slept.",
                    "The judges shouted.",
                    "The actor slept.",
                    "The artist slept.",
                ],
                "label": [0, 0, 1, 1],
                "z": [1, 0, 1, 0],
                "fast": [False, True, True, False],
            },
            identifier=Identifier(_name="MockDataPane", version="2.0"),
        )

        # Keep a copy of the original
        self.original_dataset = deepcopy(self.dataset)

        assert len(self.dataset) == 4


# class MockVisionTestBed:
#     def __init__(self, wrap_dataset: bool = False):
#         """[summary]
#
#         Args:
#             wrap_dataset (bool, optional): If `True`, create a
#             `mosaic.DataPanel`
#             ,
#                 otherwise create a
#                 `mosaic.core.dataformats.vision.VisionDataPane`
#                 Defaults to False.
#         """
#         cache_dir = os.path.join(tempfile.gettempdir(), "RGVisionTests")
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir)
#         self.image_paths = []
#         self.image_tensors = []
#         self.images = []
#
#         for i in range(200, 231, 10):
#             self.image_paths.append(os.path.join(cache_dir, "{}.png".format(i)))
#             self.image_tensors.append(i * torch.ones((10, 10, 3)))
#             save_image(self.image_tensors[-1], self.image_paths[-1])
#             self.images.append(RGImage(self.image_paths[-1]))
#
#         self.batch = {
#             "a": [{"e": 1}, {"e": 2}, {"e": 3}, {"e": 4}],
#             "b": ["u", "v", "w", "x"],
#             "i": self.image_paths,
#         }
#
#         if wrap_dataset:
#             self.dataset = DataPanel.load_image_dataset(self.batch, img_columns="i")
#         else:
#             self.dataset = VisionDataPane(self.batch, img_columns="i")
