import os
import json
from PIL import Image
import pycocotools.mask as mask_util

def prepare_ytvis_coco_json(json_path="./files/train.json"):

    # dict_keys(['info', 'licenses', 'videos', 'categories', 'annotations'])
    with open(json_path) as f:
        dset_dict = json.load(f)


    print(dset_dict["categories"])
    print(dset_dict["annotations"][0])
    print(dset_dict["videos"][0])

    print("len_videos: ", len(dset_dict["annotations"]))

    """
    annotations: dict_keys(['height', 'width', 'length', 'category_id', 'segmentations', 'bboxes', 'video_id', 'iscrowd', 'id', 'areas'])
    videos: dict_keys(['width', 'length', 'date_captured', 'license', 'flickr_url', 'file_names', 'id', 'coco_url', 'height'])
    """

    # Used for COCO-Style annotations
    images_list, annotations_list = [], []
    image_id, annot_id = 1, 1

    annotation_offsets = []


    for vid_dict in dset_dict["videos"]:

        video_id = vid_dict["id"]
        length = vid_dict["length"]

        print("Processing video: ", video_id, " with length: ", length)

        # Since it starts from 1.
        annotation_offsets.append(image_id)

        for frame_path in vid_dict["file_names"]:
            coco_dict = {}
            coco_dict["file_name"] = frame_path
            coco_dict["width"] = vid_dict["width"]
            coco_dict["height"] = vid_dict["height"]
            coco_dict["date_captured"] = vid_dict["date_captured"]
            coco_dict["license"] = vid_dict["license"]
            coco_dict["flickr_url"] = ""
            coco_dict["id"] = image_id
            images_list.append(coco_dict)
            image_id += 1

    print("Exported: ", image_id, " frames.")

    print(annotation_offsets)

    for annot_dict in dset_dict["annotations"]:

        """
        annotations: dict_keys(['height', 'width', 'length', 'category_id', 
        'segmentations', 'bboxes', 'video_id', 'iscrowd', 'id', 'areas'])
        """
        # print(annot_dict)

        num_frames = len(annot_dict["segmentations"])
        video_id = annot_dict["video_id"]

        offset = annotation_offsets[video_id - 1]

        for i in range(num_frames):

            seg = annot_dict["segmentations"][i]

            if seg is not None:

                coco_annot_dict = {}

                coco_annot_dict["image_id"] = i + offset

                coco_annot_dict["id"] = annot_id

                coco_annot_dict["iscrowd"] = annot_dict["iscrowd"]
                coco_annot_dict["height"] = annot_dict["height"]
                coco_annot_dict["width"] = annot_dict["width"]
                coco_annot_dict["category_id"] = annot_dict["category_id"]
                coco_annot_dict["area"] = annot_dict["areas"][i]
                coco_annot_dict["bbox"] = annot_dict["bboxes"][i]

                # Make it RLE compressed. frPyObjects( [pyObjects], h, w )
                compressed_rle = mask_util.frPyObjects(seg, annot_dict["height"], annot_dict["width"])
                compressed_rle["counts"] = compressed_rle["counts"].decode('ascii')
                coco_annot_dict["segmentation"] = compressed_rle

                annotations_list.append(coco_annot_dict)

                annot_id += 1

    dset_dict.pop("videos")
    dset_dict.pop("annotations")

    dset_dict["images"] = images_list
    dset_dict["annotations"] = annotations_list

    with open("./files/ytvis_train_coco_rle.json", "w") as f:
        json.dump(dset_dict, f)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prepare_ytvis_coco_json()