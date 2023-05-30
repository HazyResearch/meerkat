from typing import Callable, Dict, Tuple, Union

import meddlr as mr
import numpy as np
import PIL
import torch
from meddlr.data.transforms.subsample import PoissonDiskMaskFunc
from meddlr.forward.mri import SenseModel
from meddlr.transforms import RandomMRIMotion, RandomNoise
from meddlr.transforms.builtin.mri import MRIReconAugmentor
from torch import nn

import meerkat as mk

from .utils import is_url


class MRIPerturbationInference(mk.gui.html.div):
    """Visualize MRI reconstructions with perturbations.

    This interface allows you to visualize reconstructions of MRI slices
    when affected by noise and motion artifacts.
    Noise and motion artifacts are generated with reproducible random seeds
    that are unique for each scan.

    Current motion artifacts are limited to 2-shot 1D translational motion.

    This interface is based off the following paper:
        Desai et al. VORTEX: Physics-Driven Data Augmentations Using Consistency
        Training for Robust Accelerated MRI Reconstruction. MIDL 2022.
    """

    def __init__(
        self,
        df: mk.DataFrame,
        models: Dict[str, Union[str, Tuple[str, str], nn.Module, Callable]],
        acc: Tuple[int] = (2, 20, 1),
        sigma: Tuple[float] = (0.0, 1.0, 0.1),
        alpha: Tuple[float] = (0.0, 1.0, 0.1),
        seed: int = 0,
        device: str = "cpu",
        title: str = "MRI Perturbation Inference",
        description: str = "Visualize MRI reconstructions with perturbations.",
    ):
        """
        Args:
            df: A DataFrame containing the slices to visualize. Columns expected:
                * ``id``: The id of the scan. This should be unique to each scan.
                * ``sl``: The slice index. These should be zero-indexed and sequential.
                * ``kspace``: The fully sampled kspace for the slice.
                  Shape: (H, W, #coils)
                * ``maps``: The sensitivity maps for the slice.
                  Shape: (H, W, #coils)
                * ``target`` (optional): The target image for the slice. Shape: (H, W).
            models: A dictionary mapping model names to model URLs.
            acc: The acceleration factor range. Format: (min, max, step)
            sigma: The noise standard deviation range. Format: (min, max, step)
            alpha: The motion standard deviation range. Format: (min, max, step)
            seed: The random seed to use. This is required if results
                are to be cached.
            device: The device to use for inference.
            title: The title of the interface.
            description: The description for the interface.
        """
        super().__init__(slots=[])

        self.df = df
        self.models = models
        self.acc = acc
        self.sigma = sigma
        self.alpha = alpha
        self.seed = seed
        self.device = device
        self.title = title
        self.description = description

        # TODO: Download all models from the urls.
        view = self.build()
        self.append(view)

    def build(self):
        """Build the graph for the interface."""
        self.df.mark()

        scan_names = list(self.df["id"].unique())
        scan_name = mk.Store(scan_names[0])

        model_names = list(self.models.keys())
        model_name = mk.Store(model_names[0])

        # Acceleration factor.
        min_acc, max_acc, step_acc = self._get_range_fields(self.acc)
        acc = mk.Store(min_acc.value)
        # Noise standard deviation.
        min_sigma, max_sigma, step_sigma = self._get_range_fields(self.sigma)
        sigma = mk.Store(min_sigma.value)
        # Motion standard deviation.
        min_alpha, max_alpha, step_alpha = self._get_range_fields(self.alpha)
        alpha = mk.Store(min_alpha.value)

        # Filter the dataframe by the selected scan.
        df = self.get_scan_df(self.df, scan_name)
        max_slices = mk.len(df)
        sl = max_slices // 2
        df = self.get_slice_df(df, sl)

        # Get kspace, maps and target.
        kspace, maps, target = _get_fields(df)

        with mk.magic():
            shape = kspace.shape

        # Generate mask
        mask = self.generate_mask(shape, acc=acc)

        # Perturb kspace
        kspace = self.perturb(kspace, maps, mask=mask, sigma=sigma, alpha=alpha)

        # Zero-filled reconstruction.
        zf_image = self.zero_filled(kspace, maps, mask=mask)

        # Model reconstruction.
        model = self.load_model(model_name)
        pred = self.run_inference(kspace, maps, mask, model)

        # Create dataframe.
        df = create_df(zf_image, pred, target=target)

        # Initialize sliders.
        @mk.endpoint()
        def on_change(store: mk.Store, value):
            store.set(value)

        sl = mk.gui.Slider(
            value=sl.value,
            min=0,
            max=max_slices - 1,
            step=1,
            description="Slice",
            on_change=on_change.partial(sl),
        )
        acc = mk.gui.Slider(
            value=acc.value,
            min=min_acc,
            max=max_acc,
            step=step_acc,
            description="Acceleration",
            on_change=on_change.partial(acc),
        )
        noise = mk.gui.Slider(
            value=sigma.value,
            min=min_sigma,
            max=max_sigma,
            step=step_sigma,
            description="Noise",
            on_change=on_change.partial(sigma),
        )
        motion = mk.gui.Slider(
            value=alpha.value,
            min=min_alpha,
            max=max_alpha,
            step=step_alpha,
            description="Motion",
            on_change=on_change.partial(alpha),
        )

        zf_gallery = mk.gui.Gallery(
            df,
            main_column="Zero-Filled",
            allow_selection=True,
            cell_size=40,
        )
        gallery = mk.gui.Gallery(
            df,
            main_column="Reconstruction",
            allow_selection=True,
            cell_size=40,
        )

        # Model type
        selector = mk.gui.Select(values=model_names, value=model_name)

        # Scan
        scan_selector = mk.gui.Select(values=scan_names, value=scan_name)

        # Text.
        text = [
            mk.gui.Markdown(
                self.title,
                classes="font-bold text-slate-600 text-sm",
            )
            if self.title
            else None,
            mk.gui.Text(
                self.description,
                classes="text-slate-600 text-sm",
            )
            if self.description
            else None,
        ]

        # Overview Panel
        overview_panel = mk.gui.html.flexcol(
            [
                *text,
                mk.gui.html.grid(
                    [mk.gui.Text("Image slice", classes="text-slate-600 text-sm"), sl],
                ),
                mk.gui.html.grid(
                    [
                        mk.gui.Text("Acceleration", classes="text-slate-600 text-sm"),
                        acc,
                    ],
                ),
            ],
            classes="justify-items-start mx-4 gap-1",
        )

        # Slider Panel
        sliders = mk.gui.html.flexcol(
            [
                mk.gui.html.gridcols2(
                    [
                        mk.gui.html.grid(
                            [
                                mk.gui.Markdown(
                                    "Scan", classes="text-slate-600 text-sm"
                                ),
                                scan_selector,
                            ]
                        ),
                        mk.gui.html.grid(
                            [
                                mk.gui.Markdown(
                                    "Model", classes="text-slate-600 text-sm"
                                ),
                                selector,
                            ]
                        ),
                    ],
                    classes="gap-x-4",
                ),
                mk.gui.html.grid(
                    [mk.gui.Markdown("Noise", classes="text-slate-600 text-sm"), noise]
                ),
                mk.gui.html.grid(
                    [mk.gui.Text("Motion", classes="text-slate-600 text-sm"), motion],
                ),
            ],
            classes="items-stretch justify-items-start gap-x-4 gap-y-4 justify-content-space-between",  # noqa: E501
        )

        view = mk.gui.html.div(
            [
                mk.gui.html.grid(
                    [overview_panel, sliders],
                    classes="grid grid-cols-[1fr_1fr] space-x-5",
                ),
                mk.gui.html.grid(
                    [zf_gallery, gallery],
                    classes="grid grid-cols-[1fr_1fr] space-x-5",
                ),
            ],
            classes="gap-4 h-screen grid grid-rows-[auto_1fr] bg-white",
        )
        return view

    def _get_range_fields(self, x):
        """Get the min, max and step from a tuple."""
        if len(x) == 2:
            min_x, max_x = x
            step_x = (max_x - min_x) / 10
        else:
            min_x, max_x, step_x = x

        return mk.Store(min_x), mk.Store(max_x), mk.Store(step_x)

    @mk.reactive(backend_only=True)
    def get_scan_df(self, df: mk.DataFrame, scan_id: str) -> mk.DataFrame:
        """Get the DataFrame containing the slices to visualize."""
        return df[df["id"] == scan_id]

    @mk.reactive(backend_only=True)
    def get_slice_df(self, scan_df: mk.DataFrame, sl: int) -> mk.DataFrame:
        """Get the DataFrame containing the slices to visualize."""
        return scan_df[scan_df["sl"] == sl]

    @mk.reactive(backend_only=True)
    def load_model(self, name: str):
        """Load the model from the dictionary.

        Args:
            name: The name of the model to load.

        Returns:
            The loaded model.
        """
        model_init = self.models[name]
        if is_url(model_init):
            # Only supports huggingface format for meddlr models.
            model = mr.get_model_from_zoo(
                model_init + "/config.yaml", model_init + "/model.ckpt"
            )
        elif isinstance(model_init, (tuple, list)):
            model = mr.get_model_from_zoo(model_init[0], model_init[1])
        elif isinstance(model_init, nn.Module):
            model = model_init

        if isinstance(model, nn.Module):
            model = model.to(self.device).eval()
        return model

    @mk.reactive(backend_only=True)
    def run_inference(self, kspace, maps, mask: torch.Tensor, model):
        with torch.no_grad():
            outputs = model({"kspace": kspace, "maps": maps, "mask": mask})
        return outputs["pred"]

    @mk.reactive(backend_only=True)
    def generate_mask(self, shape, acc: float):
        """Generate Poisson Disc undersampling mask.

        To generate other types of masks, see
        :mod:`meddlr.data.transforms.subsample`.
        """
        shape = shape[:-1]
        mask_func = PoissonDiskMaskFunc(accelerations=(acc,), calib_size=24)
        mask = mask_func(shape, seed=self.seed)
        mask = mask.unsqueeze(-1)
        return mask

    @mk.reactive(backend_only=True)
    def perturb(
        self,
        kspace: torch.Tensor,
        maps: torch.Tensor,
        mask: torch.Tensor,
        sigma: float,
        alpha: float,
    ) -> torch.Tensor:
        """Add perturbations to kspace.

        This method 1) undersamples the kspace and adds 2) motion and
        3) noise artifacts.

        TODO: Add support for other types of perturbations.

        Args:
            kspace: The fully sampled kspace.
            maps: The sensitivity maps.
            mask: The undersampling mask.

        Returns:
            torch.Tensor: The perturbed kspace.
        """
        kspace = kspace * mask
        # Scale sigma relatively to kspace.
        sigma = sigma * torch.quantile(kspace.abs(), 0.999)
        noise = RandomNoise(p=1.0, std_devs=(sigma, sigma), use_mask=True).seed(
            self.seed
        )
        motion = RandomMRIMotion(p=1.0, std_devs=alpha).seed(self.seed)
        tfms = [motion if alpha > 0 else None, noise if sigma > 0 else None]
        tfms = [tfm for tfm in tfms if tfm is not None]
        if tfms:
            augmentor = MRIReconAugmentor(tfms, seed=self.seed)
            outputs, _, _ = augmentor(
                kspace=kspace,
                maps=maps,
                mask=mask,
            )
            kspace = outputs["kspace"]
        return kspace

    @mk.reactive(backend_only=True)
    def zero_filled(self, kspace: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor):
        A = SenseModel(maps=maps, weights=mask)
        out = A(kspace, adjoint=True)
        return out


@mk.reactive()
def to_pil(image: torch.Tensor) -> PIL.Image:
    """Convert a torch tensor to a PIL image.

    Args:
        image: The image to convert.
            If RGB, last dimension should be the channel dimension.

    Returns:
        The converted PIL image.
    """
    image = image.squeeze().abs()
    image = image - image.min()

    scale = image.max()  # torch.quantile(image, 0.95)
    image = torch.clamp(image / scale, 0, 1)
    image = image.cpu().numpy()

    image = (image * 255).round().astype(np.uint8)
    image = np.stack([image] * 3, axis=-1)

    image = PIL.Image.fromarray(image)
    # if image.mode != 'RGB':
    #     image = image.convert('RGB')
    return image


@mk.reactive()
def create_df(zf_image, pred, target=None):
    df = mk.DataFrame(
        {
            "Zero-Filled": [to_pil(zf_image)],
            "Reconstruction": [to_pil(pred)],
            "Target": [to_pil(target)] if target is not None else [None],
        }
    )
    return df


@mk.reactive(backend_only=True, nested_return=True)
def _get_fields(df: mk.DataFrame):
    """Get a row from a dataframe."""
    row = df[["kspace", "maps", "target"]]._get(0, materialize=True)
    # Add batch dimension.
    return (
        row["kspace"].unsqueeze(0),
        row["maps"].unsqueeze(0),
        row["target"].unsqueeze(0),
    )
