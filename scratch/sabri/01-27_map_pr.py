import meerkat as mk
import torch
import clip
from PIL import Image


df = mk.get("imagenette")[:100]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@torch.no_grad()
def embed_image(img: torch.tensor):
    return model.encode_image(img.unsqueeze(0).to(device)).cpu().detach().numpy().squeeze()

df["preprocessed"] = df.defer(
    lambda img: preprocess(img)
)

df["emb"] = df.map(
    lambda preprocessed: embed_image(preprocessed),
    use_ray=True
)

breakpoint()
