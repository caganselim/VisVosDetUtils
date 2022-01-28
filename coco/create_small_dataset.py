import json
import os
import shutil

coco_root = "/home/kuavg/datasets/coco/"

export_root = "./tiny_coco"
if os.path.exists(export_root):
    shutil.rmtree(export_root)
os.mkdir(export_root)


json_file_in = "./files/coco_train2017_rle.json"
json_file_out = "instances_train2017.json"


# This is list of IDs, if sample_new_images variable is False, we'll use these.
images_list = [391895, 522418, 184613, 318219, 554625, 574769, 60623, 309022,
               5802, 222564, 118113, 193271, 224736, 483108, 403013, 374628]

with open(json_file_in, 'r') as f:
    coco_dict = json.load(f)

sample_new_images = False
num = 64

if sample_new_images:
    images_list = [item['id'] for item in coco_dict['images'][:num]]
    print(images_list)

# Export annotations
annot_save_pth = os.path.join(export_root, json_file_out)
new_images = []
new_annotations = []

# scan images
for img in coco_dict['images']:
    if img['id'] in images_list:
        new_images.append(img)

# scan annotations
for ann in coco_dict['annotations']:
    if ann['image_id'] in images_list:
        new_annotations.append(ann)

# save
coco_dict['images'] = new_images
coco_dict['annotations'] = new_annotations

with open(annot_save_pth, 'w') as ff:
    json.dump(coco_dict, ff)

# Export images
new_dir_path = os.path.join(export_root, "train2017")
if os.path.exists(new_dir_path):
    shutil.rmtree(new_dir_path)
os.mkdir(new_dir_path)

for img in images_list:
    img_name = format(img, "012") + ".jpg"
    src_file  = os.path.join(coco_root, "train2017", img_name)
    dest_file = os.path.join(new_dir_path, img_name)
    shutil.copyfile(src_file, dest_file)
