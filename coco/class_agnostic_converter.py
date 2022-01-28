import json

in_json = "./instances_tiny_coco.json"
out_json = "instances_val2017_class_agnostic.json"

with open(in_json) as f:

    """
    This function converts a std COCO dataset to a class-agnostic one.
    """

    # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    annot_dict = json.load(f)

    annot_dict['categories'] = [{'supercategory': 'object', 'id': 1, 'name': 'object'}]

    for idx, record in enumerate(annot_dict['annotations']):

        # Set this to one.
        annot_dict['annotations'][idx]['category_id'] = 1


    with open(out_json, 'w') as json_file:
        json.dump(annot_dict, json_file)
