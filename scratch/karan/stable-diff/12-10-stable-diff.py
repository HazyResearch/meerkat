import meerkat as mk
import argparse
import os
import time

from meerkat.interactive.formatter import ImageURLFormatter

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/data/datasets/opendata/")

ngoa = mk.get(
    "ngoa",
    "/Users/krandiash/Desktop/workspace/projects/datasci/data/ngoa/opendata/",
)
ngoa_images = ngoa["published_images"][:200]
ngoa_images["image_1000_url"] = ngoa_images["iiifurl"].map(
    lambda x: f"{x}/full/!1000,1000/0/default.jpg", pbar=True
)
ngoa_images["image_1000_url"].formatter = ImageURLFormatter()

ngoa_images["image_224_url"] = ngoa_images["iiifurl"].map(
    lambda x: f"{x}/full/224,224/0/default.jpg", pbar=True
)
ngoa_images["image_224"] = mk.ImageColumn.from_urls(ngoa_images["image_224_url"])
ngoa_images["clip(image_224)"] = mk.DataFrame.read("ngoa-224-n200-clip.mk")[
    "clip(image_224)"
]


# Make the clients for the two APIs
# os.environ["TOMA_SD_URL"] = "https://api.together.xyz"
# sd_client = Manifest(
#     client_name="toma-sd",
# )

# os.environ["TOMA_SD_URL"] = "https://zentrum.xzyao.dev"
# sd_img2img_client = Manifest(
#     client_name="toma-sd",
# )


ngoa_images = ngoa_images[["image_1000_url", "image_224_url", "clip(image_224)"]]

match = mk.gui.Match(df=ngoa_images, against="image_224", col="image_1000_url")
sorted_df = mk.sort(ngoa_images, by=match.col, ascending=False)


@mk.gui.reactive
def subselect_columns(df, columns):
    return df[columns]


gallery = mk.gui.Gallery(
    df=subselect_columns(sorted_df, ["image_1000_url"]),
    main_column="image_1000_url",
    tag_columns=[],
    primary_key="image_1000_url",
)


# Store results of image to image generation
sdimg2img_df = mk.DataFrame(
    {
        "url": [],  # URL where the image is stored
        "prev_url": [],  # URL of the base image
        "prompt": [],  # Prompt used to generate the image
        "strength": [],
        "n": [],
        "id": [],
        "run_id": [],
        "time": [],
    }
)
sdimg2img_df["url"].formatter = ImageURLFormatter()

# Store results of text to image generation
sd_df = mk.DataFrame(
    {
        "url": [],
        "prompt": [],
        "n": [],
        "id": [],
        "run_id": [],
        "time": [],
        "strength": [],
        "prev_url": [],
    }
)
sd_df["url"].formatter = ImageURLFormatter()

gen_df = mk.DataFrame({k: [] for k in sd_df.columns})


@mk.gui.endpoint(prefix="/stable-diffusion")
def run_sd_generation(gen_df: mk.DataFrame, prompt: mk.gui.Store[str], n: mk.gui.Store[int]):
    """
    Run Stable Diffusion generation on the prompt and return the results.
    """
    # Generate the image
    start = time.time()
    # res = sd_client.run(prompt, n=n)
    import replicate
    os.environ["REPLICATE_API_TOKEN"] = "f618371c61e6bb05c4c5624e4ff38f5068428814"
    model = replicate.models.get("stability-ai/stable-diffusion")
    version = model.versions.get("6359a0cab3ca6e4d3320c33d79096161208e9024d174b2311e5a21b6c7e1131c")
    res = version.predict(prompt=prompt)
    end = time.time()

    if not isinstance(res, list):
        res = [res]

    # Store the results
    result_df = mk.DataFrame(
        {
            "url": res,
            "prompt": [prompt.value] * len(res),
            "n": [n.value] * len(res),
            "id": [e.split("/")[-1] for e in res],
            "run_id": [None] * len(res),
            "time": [end - start] * len(res),
        }
    )
    result_df["strength"] = result_df["id"].map(lambda x: None)
    result_df["prev_url"] = result_df["id"].map(lambda x: None)
    result_df["url"].formatter = ImageURLFormatter()

    # Store the results
    if len(gen_df) == 0:
        gen_df.set(result_df)
    else:
        gen_df.set(mk.concat([gen_df, result_df], axis=0))

    print(gen_df)
    print(gen_df.shape)


@mk.gui.endpoint
def run_sd_img2img_generation(prompt: str, strength: float, n: int, selected: list):
    # Create a default return value
    result_df = mk.DataFrame({k: [] for k in sd_df.columns})
    result_df["url"].formatter = ImageURLFormatter()

    if len(selected) != 1:
        # TODO: Allow users in the gallery to only select one image
        return result_df

    if prompt:
        # Generate the image
        start = time.time()
        result = sd_img2img_client.run(
            prompt,
            engine="SDImg2Img",
            n=n,
            url=selected[0],
            strength=strength,
        )
        end = time.time()
        if n == 1:
            result = [result]

        # Store the results
        result_df = mk.DataFrame(
            {
                "url": result,
                "prompt": [prompt] * len(result),
                "n": [n] * len(result),
                "id": [res.split("/")[-1] for res in result],
                "run_id": [sd_img2img_client.client.get_last_job_id()] * len(result),
                "time": [end - start] * len(result),
            }
        )
        result_df["url"].formatter = ImageURLFormatter()

        mk.concat([gen_df, result_df], axis=0)


textbox = mk.gui.Textbox(text="", title="Stable Diffusion Prompt")
strength = mk.gui.Choice(value=0.5, choices=[0.1, 0.3, 0.5, 0.7, 0.9])
n = mk.gui.Choice(value=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sd_button = mk.gui.Button(
    title="Generate",
    # Run the `run_sd_generation` endpoint when the button is clicked
    on_click=run_sd_generation.partial(gen_df, textbox.text, n.value),
)
sdimg2img_button = mk.gui.Button(
    title="Generate (Img2Img)",
    # Run the `run_sd_img2img_generation` endpoint when the button is clicked
    on_click=run_sd_img2img_generation.partial(
        textbox.text,
        strength.value,
        n.value,
        gallery.selected,
    ),
)
generation_gallery = mk.gui.Gallery(
    df=gen_df,
    main_column="url",
    tag_columns=[],
)

mk.gui.start()
mk.gui.Interface(
    component=mk.gui.RowLayout(
        components=[
            match,
            gallery,
            textbox,
            strength,
            n,
            sd_button,
            sdimg2img_button,
            generation_gallery,
        ]
    )
).launch()
