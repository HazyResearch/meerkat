import meerkat as mk

df = mk.get("imagenette")

from PIL.Image import Image
import numpy as np


class ParachuteClassifier:
    def preprocess(self, img: Image) -> np.ndarray:
        """Prepare an image for classification."""
        return np.array(img.convert("RGB").resize((224, 224)))

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Classify a batch of images as containing a parachute or not."""
        return batch[:, :, :, 2].mean(axis=1).mean(axis=1) > 0.5


classifier = ParachuteClassifier()

preprocessed = df["img"].defer(classifier.preprocess)
df["prediction"] = preprocessed.map(
    classifier.predict, is_batched_fn=True, batch_size=32
)
