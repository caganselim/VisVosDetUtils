import os
from tqdm import tqdm
import json

"""
Use this script to delete the files.
"""

subset_to_delete = "train_f"

files = os.listdir(subset_to_delete)


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


with open("/home/kuavg/Desktop/det100trainv2.json") as f:
    dataset = json.load(f)

ims = {}

for im in dataset["images"]:

    if im["dataset"] != "openimages":
        continue

    l = im["file_name"].split('/')

    # print(im["file_name"], l)

    subset, filename = l[0], l[1]

    if subset == subset_to_delete:
        ims[filename] = ""

print(len(ims.keys()))

for f in tqdm(files):
    if not (f in ims):
        path = os.path.join(subset_to_delete, f)
        remove(path)