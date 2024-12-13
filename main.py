import os, sys
import json
import time
from argparse import ArgumentParser

# pip install -U "ray[default]"
import ray

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def read_image(img_path: str) -> np.ndarray:
    img = Image.open(img_path)
    img = np.asarray(img)
    return img.astype(np.float32) / 255


def save_image(img, path, suffix) -> None:
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)

    # parse destination path
    base, name = path.split('\\')
    name = name[:-len('.jpg')]

    # create output directory
    os.makedirs(f"out{base}", exist_ok=True)

    # save image
    img.save(f"out{base}/{name}{suffix}.jpg")


def generate_preblurred_image(img: np.ndarray, max_value: int):
    """
    Generate a pre-blurred image for all levels
    :param img:
    :param max_value:
    :return:
    """
    levels = np.empty((max_value + 1, *img.shape), dtype=np.float32)  # Pre-allocate memory
    levels[0] = img   # Store the original image in the first level

    # Apply Gaussian filters for all levels
    for i in range(max_value):
        sigma = i + 1
        gaussian_filter(img, sigma=(sigma, sigma, 0), output=levels[i+1], truncate=2)  # Store directly in `levels`

    return levels


@ray.remote
def task(img_path: str, sigma_map: np.ndarray, max_level: int, suffix: str, correct_source_images: bool = False) -> None:
    # global sigma_map_shared
    print("Starting", img_path)
    start = time.time()
    img = read_image(img_path)
    # blurred_img = img
    # only compute it for the right side of the image (the last 60%)
    # img_orig = img.copy()
    # orig_H, orig_W, _ = img_orig.shape
    # img = img_orig[:, int(0.4 * orig_W):]
    # sigma_map = sigma_map[:, int(0.4 * orig_W):]

    # generate image stack with increasing blur levels
    levels = generate_preblurred_image(img, max_level)

    # compute the two levels to interpolate between
    left = np.floor(sigma_map).astype(int)
    sigma_offset = sigma_map - left
    N, H, W, C = levels.shape

    # interpolate between the two levels
    blurred_img = levels[left, np.arange(H)[:, None], np.arange(W)[None, :], :] * (1 - sigma_offset[..., None]) + levels[left+1, np.arange(H)[:, None], np.arange(W)[None, :], :] * sigma_offset[..., None]

    # restore the dynamic range
    # blurred_img = np.clip((blurred_img - 2/255) * 1/(152/255), 0, 1)
    blurred_img = np.clip((blurred_img - 4/255) * 1/(174/255), 0, 1)
    save_image(blurred_img, img_path, suffix)

    if correct_source_images:
        # img = np.clip((img - 2/255) * 1/(152/255), 0, 1)
        img = np.clip((img - 4/255) * 1/(174/255), 0, 1)
        save_image(img, img_path, "-s0")


    # restore original image size
    # img_orig[:, int(0.4 * orig_W):] = blurred_img
    # blurred_img = img_orig

    print("Done", img_path, "in", time.time() - start, "secs")


def parse_args():
    parser = ArgumentParser(description="Foveated Blurring")
    parser.add_argument("--sigma_map", "-sm", type=str, help="Path to the sigma map", required=True)
    parser.add_argument("--source", '-src', type=str, help="Path to the source images", required=True)
    parser.add_argument("--suffix", '-suf', type=str, help="Suffix to add to the output images", default="-s20")
    parser.add_argument("--correct_source", '-s0', help="Path to the source images", action="store_true")
    parser.add_argument("--num_processes", "-np", type=int, help="Number of processes to use", default=9)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # pretty print the arguments
    print(json.dumps(vars(args), indent=4))

    sigma_map_path = "SigmaPX-20.png"
    max_level = int(sigma_map_path.split('-')[1].split('.')[0])
    sigma_map = read_image(args.sigma_map) * max_level #* (5760 / (112.62 * 60))

    start = time.time()

    ray.init(num_cpus=args.num_processes)
    tasks = []
    path = args.source
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img_path = os.path.join(path, file)
            tasks.append(task.remote(img_path, sigma_map, max_level, args.suffix, args.correct_source))

    ray.get(tasks)
    print("All done in", time.time() - start, "secs")