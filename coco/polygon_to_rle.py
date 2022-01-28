import json
import pycocotools.mask as mask_util

def polygons_to_bitmask(polygons,  height, width, iscrowd):
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return []
    rles = mask_util.frPyObjects(polygons, height, width)
    if iscrowd:
        rle = rles
    else:
        rle = mask_util.merge(rles)

    return rle


with open("./instances_val2017.json") as f:
    dset = json.load(f)
print(dset.keys())

# Keys are:
# dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

# Cache them first
img_sizes = {}
for img in dset["images"]:
    h = img["height"]
    w = img["width"]
    image_id = img["id"]
    img_sizes[image_id] = (h,w)

for idx, annot in enumerate(dset["annotations"]):
    h, w =  img_sizes[annot["image_id"]]

    compressed_rle = polygons_to_bitmask(annot["segmentation"], h, w, annot["iscrowd"])
    compressed_rle["counts"] = compressed_rle["counts"].decode('ascii')
    annot["segmentation"] = compressed_rle    
    dset["annotations"][idx] = annot


with open('files/coco_val2017_rle.json', 'w') as json_file:
    json.dump(dset, json_file)
