import torchvision.transforms as transforms
from torchvision import models

from meerkat.columns.image_column import ImageColumn
from meerkat.datapanel import DataPanel
from meerkat.model.tensormodel import TensorModel


def test_segmentation():
    dp = DataPanel(
        {
            "img": ImageColumn.from_filepaths(
                ["/home/priya/ws-rg/bird.png", "/home/priya/ws-rg/bird.png"]
            ),
            "filepath": ["/home/priya/ws-rg/bird.png", "/home/priya/ws-rg/bird.png"],
        }
    )
    print(dp.head())
    print(dp.columns)
    print("----")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dp["input"] = ImageColumn.from_filepaths(
        filepaths=dp["filepath"],
        transform=transform,
    )
    print(dp.head())
    print(dp.columns)
    print("----")

    fcn = models.segmentation.fcn_resnet101(pretrained=True)
    tm = TensorModel(model=fcn, task="semantic_segmentation")
    print(tm.model, tm.device, tm.task, tm.is_classifier)

    out = tm.output(dp, input_columns=["input"], batch_size=1)
    print(out.columns)
    print(out["logits"].num_classes, out["logits"].multi_label)
    print(out["logits"][0])
    mask = out["logits"].binarymask(class_index=2)
    print(type(mask))
    print(mask[0].shape)
    print(mask[0])
    print(out["preds"][0])


def test_tensor_classification():
    import os

    import torchvision.transforms as transforms
    from torchvision.models import resnet18

    model = resnet18(pretrained=True)

    from mosaic.columns.image_column import ImageColumn
    from mosaic.contrib.imagenette import download_imagenette

    BASE_DIR = "/home/priya/datasets"
    dataset_dir = download_imagenette(BASE_DIR)
    # 1. Create `DataPanel`
    dp = DataPanel.from_csv(os.path.join(dataset_dir, "imagenette.csv"))
    # 2. Create `ImageColumn`
    dp["img"] = ImageColumn.from_filepaths(filepaths=dp["img_path"])
    print(dp.head())

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 2. Create new column with transform

    valid_dp = dp.lz[dp["split"].data == "valid"]
    valid_dp["input"] = ImageColumn.from_filepaths(
        filepaths=valid_dp["img_path"],
        transform=transform,
    )

    tm = TensorModel("test_model", model)

    act = tm.get_activation(valid_dp, "layer4", ["input"], 256)
    print(type(act))
    print(valid_dp.columns)
    # return cls_dp


test_segmentation()
