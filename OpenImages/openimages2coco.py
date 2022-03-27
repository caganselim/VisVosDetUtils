import glob
import json
import os
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm


with open("./csvs/images_to_skip.txt") as f:
    images_to_skip_list = f.readlines()

images_to_skip_dict = {}
for im in images_to_skip_list:
    images_to_skip_dict[im.strip()] = ""

##################################################

def txt2list(pth):
    with open(pth) as f:
        lines = f.readlines()
    lines = [l.strip().split(',') for l in lines]

    return lines


mapping_pth = "csvs/mapping.csv"
mapping_lines = txt2list(mapping_pth)

labels = [line[0].replace("/", "") for line in mapping_lines]


openimgs2ytvis = {}
for line in mapping_lines:
    openimgs2ytvis[line[0].replace("/", "")] = line[2].strip()



####################################################


ytvis_cnts = {}
with open('ytvis_cnts.csv') as f:
    for line in f:
        line = line.strip().split(',')
        cat = line[0]
        count = int(line[1])
        ytvis_cnts[cat] = count


openimgs_cnts = {}
with open('openimgs_cnts.csv') as f:
    for line in f:
        line = line.strip().split(',')
        cat = line[0]
        count = int(line[1])
        openimgs_cnts[cat] = count


fused_openimgs_cnts = {}
for label, cnt in openimgs_cnts.items():
    ytvis_cat = openimgs2ytvis[label]
    if ytvis_cat not in fused_openimgs_cnts:
        fused_openimgs_cnts[ytvis_cat] = 0
    fused_openimgs_cnts[ytvis_cat] += cnt


ytvis_classes = {}
with open("csvs/ytvis_ids.csv") as f:
    for line in f:
        line = line.strip().split(',')
        cat_id = int(line[0])
        cat = line[1]
        ytvis_classes[cat] = cat_id


folder_ids = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]

openimages2coco = {"images": [], "annotations": []}
# ids after ytvis + overlapping coco
image_id = 0
annotation_id = 1

openimages = {}
for fid in folder_ids:
    folder = "masks/train-masks-" + fid
    mask_images = os.listdir(folder)
    for idx, img in tqdm(enumerate(mask_images)):
        first_split = img.find('_')
        last_split = img.rfind('_')
        image_name = img[:first_split]
        cat_openimg = img[first_split+1:last_split]
        cat_ytvis = openimgs2ytvis.get(cat_openimg, None)
        if cat_ytvis is None:
            continue
        
        
        if image_name in images_to_skip_dict:
            continue
            
        cat_id = ytvis_classes[cat_ytvis]
        img_pth = os.path.join(folder, img)
        bin_mask = np.asfortranarray(Image.open(img_pth))
        encoded_mask = mask_util.encode(bin_mask)
        area = mask_util.area(encoded_mask)
        bbox = mask_util.toBbox(encoded_mask).astype(int).tolist()
        segm = {'size': encoded_mask['size'], 'counts': encoded_mask['counts'].decode('utf-8')}
        
        if image_name not in openimages:
            image_id += 1
            openimages[image_name] = image_id          
            openimages2coco["images"].append({
                "dataset": "openimages",
                "filename": "train_" + fid + "/" + image_name + ".jpg",
                "height": segm['size'][0],
                "width": segm['size'][1],
                "id": image_id
            }) 
        
        openimages2coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "iscrowd": 0,
            "category_id": cat_id,
            "segmentation": segm,
            "bbox": bbox,
            "area": int(area)
        })
        annotation_id += 1
    
    
    print("Finished processing folder " + folder)


with open('openimages2coco.json', 'w') as f:
    json.dump(openimages2coco, f)
