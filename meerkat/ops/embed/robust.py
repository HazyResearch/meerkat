import os
import subprocess
from typing import Dict, Union

from .encoder import Encoder
from .registry import encoders

VARIANTS = {
    # flake8: noqa
    "imagenet_l2_3_0": "https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0",
    "cifar_l2_1_0": "https://www.dropbox.com/s/s2x7thisiqxz095/cifar_l2_1_0.pt?dl=0",
    # flake8: noqa
    "imagenet_linf_8": "https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0",
}


@encoders.register
def robust(
    variant: str = "imagenet_l2_3_0",
    device: Union[int, str] = "cpu",
    model_path: str = None,
) -> Dict[str, Encoder]:
    """Image classifier trained with adversarial robustness loss.

    [engstrom_2019]_.

    Args:
        variant (str, optional): One of ["imagenet_l2_3_0", "cifar_l2_1_0",
            "imagenet_linf_8"].Defaults to "imagenet_l2_3_0".
        device (Union[int, str], optional): The device on which the encoders will be
            loaded. Defaults to "cpu".


    .. [engstrom_2019]

       @misc{robustness,
            title={Robustness (Python Library)},
            author={Logan Engstrom and Andrew Ilyas and Hadi Salman and Shibani
            Santurkar and Dimitris Tsipras},
            year={2019},
            url={https://github.com/MadryLab/robustness}
        }
    """

    model_path = (
        os.path.expanduser("~/.cache/domino/robust/robust_resnet50.pth")
        if model_path is None
        else model_path
    )
    model = _load_robust_model(model_path=model_path, variant=variant).to(device)

    return {
        "image": Encoder(
            encode=lambda x: model(x, with_latent=True)[0][1],
            preprocess=_transform_image,
        ),
    }


def _load_robust_model(model_path: str, variant: str):
    try:
        from robustness import datasets as dataset_utils
        from robustness import model_utils
    except ImportError:
        raise ImportError("To embed with robust run `pip install robustness`")

    # ensure model_path directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    subprocess.run(
        [
            "wget",
            "-O",
            model_path,
            "https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0",
        ]
    )

    dataset_function = getattr(dataset_utils, "ImageNet")
    dataset = dataset_function("")

    model_kwargs = {
        "arch": variant,
        "dataset": dataset,
        "resume_path": model_path,
        "parallel": False,
    }
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()
    return model


def _transform_image(img):
    from torchvision import transforms

    return transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )(img)
