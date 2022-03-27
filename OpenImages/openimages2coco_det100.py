import glob
import json
import os
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm
import csv

folder_ids = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
openimages2coco = {"images": [], "annotations": []}

with open("./csvs/openimages2det100.csv", newline='') as csvfile:
    map_reader = csv.reader(csvfile, delimiter=',')
    map_dict = {}
    for row in map_reader:
        open_imgs_id = row[0].replace("/", "")
        uber_id = int(row[2])
        if uber_id != -1:
            map_dict[open_imgs_id] = uber_id

print(map_dict)

with open("./csvs/images_to_skip.txt") as f:
    images_to_skip_list = f.readlines()

images_to_skip_dict = {}
for im in images_to_skip_list:
    images_to_skip_dict[im.strip()] = ""

image_id = 0
annotation_id = 1
openimages_cached = {}
for fid in folder_ids:

    folder = "train-masks-" + fid
    mask_folder = os.path.join("./masks",folder)
    mask_images = os.listdir(mask_folder)

    for idx, img in tqdm(enumerate(mask_images)):

        first_split = img.find('_')
        last_split = img.rfind('_')
        image_name = img[:first_split]
        cat_openimg = img[first_split+1:last_split]


        if cat_openimg in map_dict:
            uber_cat_id = map_dict[cat_openimg]
        else:
            continue

        if image_name in images_to_skip_dict:
            continue

        img_pth = os.path.join(mask_folder, img)
        bin_mask = np.asfortranarray(Image.open(img_pth))
        encoded_mask = mask_util.encode(bin_mask)
        area = mask_util.area(encoded_mask)
        bbox = mask_util.toBbox(encoded_mask).astype(int).tolist()
        segm = {'size': encoded_mask['size'], 'counts': encoded_mask['counts'].decode('utf-8')}

        if image_name not in openimages_cached:
            openimages_cached[image_name] = image_id
            openimages2coco["images"].append({
                "dataset": "openimages",
                "file_name": "train_" + fid + "/" + image_name + ".jpg",
                "height": segm['size'][0],
                "width": segm['size'][1],
                "id": image_id
            })
            image_id += 1
        openimages2coco["annotations"].append({
            "id": annotation_id,
            "image_id": openimages_cached[image_name],
            "iscrowd": 0,
            "category_id": uber_cat_id,
            "segmentation": segm,
            "bbox": bbox,
            "area": int(area)
        })
        annotation_id += 1

    print("Finished processing folder " + folder)

with open('openimages2det100.json', 'w') as f:
    json.dump(openimages2coco, f)
