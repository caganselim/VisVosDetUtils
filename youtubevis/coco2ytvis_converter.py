import csv

def json_read(pth):
    with open(pth) as f:
        json_dict = json.load(f)
    return json_dict


mapping_dict = {}
with open("./coco2ytvis.csv", newline='') as csvfile:
    map_reader = csv.reader(csvfile, delimiter=',')
    for row in map_reader:
        if row[3]: 
            mapping_dict[int(row[0])] = int(row[3])
            
            
print(mapping_dict)


import json

coco_dict = json_read("coco_train2017_rle.json")

annot_out = []
images_out = []
image_ids = {}

for annotation in coco_dict["annotations"]:

    category_id = annotation["category_id"]
    if category_id in mapping_dict:
        cat_out = mapping_dict[category_id]
        annotation["category_id"] = cat_out
        annot_out.append(annotation)
        
        image_id = annotation["image_id"]
        
        if image_id not in image_ids:
            image_ids[image_id] = ""


for image in coco_dict["images"]:

    id = image["id"]
    
    if id in image_ids:
        image["dataset"] = "coco2017train" # VERY IMPORTANT
        images_out.append(image)
        
        
print(len(images_out))
print(len(annot_out))

coco_dict["annotations"] = annot_out
coco_dict["images"] = images_out
coco_dict["categories"] = [{"supercategory": "object", "id": 1, "name": "person"}, {"supercategory": "object", "id": 2, "name": "giant_panda"}, {"supercategory": "object", "id": 3, "name": "lizard"}, {"supercategory": "object", "id": 4, "name": "parrot"}, {"supercategory": "object", "id": 5, "name": "skateboard"}, {"supercategory": "object", "id": 6, "name": "sedan"}, {"supercategory": "object", "id": 7, "name": "ape"}, {"supercategory": "object", "id": 8, "name": "dog"}, {"supercategory": "object", "id": 9, "name": "snake"}, {"supercategory": "object", "id": 10, "name": "monkey"}, {"supercategory": "object", "id": 11, "name": "hand"}, {"supercategory": "object", "id": 12, "name": "rabbit"}, {"supercategory": "object", "id": 13, "name": "duck"}, {"supercategory": "object", "id": 14, "name": "cat"}, {"supercategory": "object", "id": 15, "name": "cow"}, {"supercategory": "object", "id": 16, "name": "fish"}, {"supercategory": "object", "id": 17, "name": "train"}, {"supercategory": "object", "id": 18, "name": "horse"}, {"supercategory": "object", "id": 19, "name": "turtle"}, {"supercategory": "object", "id": 20, "name": "bear"}, {"supercategory": "object", "id": 21, "name": "motorbike"}, {"supercategory": "object", "id": 22, "name": "giraffe"}, {"supercategory": "object", "id": 23, "name": "leopard"}, {"supercategory": "object", "id": 24, "name": "fox"}, {"supercategory": "object", "id": 25, "name": "deer"}, {"supercategory": "object", "id": 26, "name": "owl"}, {"supercategory": "object", "id": 27, "name": "surfboard"}, {"supercategory": "object", "id": 28, "name": "airplane"}, {"supercategory": "object", "id": 29, "name": "truck"}, {"supercategory": "object", "id": 30, "name": "zebra"}, {"supercategory": "object", "id": 31, "name": "tiger"}, {"supercategory": "object", "id": 32, "name": "elephant"}, {"supercategory": "object", "id": 33, "name": "snowboard"}, {"supercategory": "object", "id": 34, "name": "boat"}, {"supercategory": "object", "id": 35, "name": "shark"}, {"supercategory": "object", "id": 36, "name": "mouse"}, {"supercategory": "object", "id": 37, "name": "frog"}, {"supercategory": "object", "id": 38, "name": "eagle"}, {"supercategory": "object", "id": 39, "name": "earless_seal"}, {"supercategory": "object", "id": 40, "name": "tennis_racket"}]


with open("coco_train2017_ytvis_classes.json", "w") as f:
    json.dump(coco_dict, f)




