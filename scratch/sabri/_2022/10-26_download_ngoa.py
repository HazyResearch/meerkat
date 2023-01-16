import meerkat as mk
import torch.multiprocessing as mp


dataset = mk.get("ngoa")

if __name__ == "__main__":
    mp.freeze_support()
    dataset["published_images"]["image_224"].map(lambda x: True, batch_size=4, num_workers=12, pbar=True)
