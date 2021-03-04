import os
import os.path as path
from collections import namedtuple

import numpy as np
import pandas as pd
from PIL import Image

import torch as t

DATAROOT = path.expanduser("~/ml-data/plant-pathology")
Stats = namedtuple("Stats", ["means", "stds"])


def resize_images(height, width):
    cache_root = path.join(DATAROOT, "cache", f"{height}x{width}")
    if path.exists(cache_root):
        raise RuntimeError(
            f"{cache_root} already exists. Delete the directory and its contents before proceeding."
        )
    os.mkdir(cache_root)

    size = (width, height)
    imgroot = path.join(DATAROOT, "images")
    for img_file in os.listdir(imgroot):
        img_path = path.join(imgroot, img_file)
        resized_img = Image.open(img_path).resize(size)
        resized_img_path = path.join(cache_root, img_file)
        resized_img.save(resized_img_path)


def calc_meta(imgroot):
    train_df = pd.read_csv(path.join(DATAROOT, "train.csv"))

    sum_red, sum_red_2 = 0, 0
    sum_green, sum_green_2 = 0, 0
    sum_blue, sum_blue_2 = 0, 0
    num_pixels = 0

    for row in train_df.itertuples():
        img_path = path.join(imgroot, f"{row.image_id}.jpg")
        img = Image.open(img_path)
        reds = np.asarray(img.getdata(band=0))
        sum_red += np.sum(reds)
        sum_red_2 += np.sum(reds ** 2)

        greens = np.asarray(img.getdata(band=1))
        sum_green += np.sum(greens)
        sum_green_2 += np.sum(greens ** 2)

        blues = np.asarray(img.getdata(band=2))
        sum_blue += np.sum(blues)
        sum_blue_2 += np.sum(blues ** 2)

        num_pixels += len(reds)

    mean_red = sum_red / num_pixels
    std_red = np.sqrt((sum_red_2 / num_pixels) - (mean_red ** 2))

    mean_green = sum_green / num_pixels
    std_green = np.sqrt((sum_green_2 / num_pixels) - (mean_green ** 2))

    mean_blue = sum_blue / num_pixels
    std_blue = np.sqrt((sum_blue_2 / num_pixels) - (mean_blue ** 2))

    return Stats(
        means=(mean_red / 255.0, mean_green / 255.0, mean_blue / 255.0),
        stds=(std_red / 255.0, std_green / 255.0, std_blue / 255.0),
    )


def restore_image(img, means, stds):
    rescaled_red = ((img[0] * stds[0]) + means[0]) * 255
    rescaled_green = ((img[1] * stds[1]) + means[1]) * 255
    rescaled_blue = ((img[2] * stds[2]) + means[2]) * 255
    rescaled_img = (
        t.stack((rescaled_red, rescaled_green, rescaled_blue))
        .permute(1, 2, 0)
        .to(t.uint8)
    )
    return Image.fromarray(rescaled_img.numpy())


if __name__ == "__main__":
    imgroot = path.join(DATAROOT, "cache", "250x250")
    stats = calc_meta(imgroot)
    print(stats)
