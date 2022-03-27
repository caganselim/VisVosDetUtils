## Script usages

* open_images_cleaner.py => OpenImages dataset is huge. If you run this script, it can clean your downloaded images whether it is included in a .json file, and you can save space :)
* images_to_skip.py => This can export a list of image-ids for which rotation info is missin in OpenImages dataset. For more info, see this link: https://storage.googleapis.com/openimages/web/2018-05-17-rotation-information.html
* openimages2coco.py => process openimages masks to a COCO format, considering YoutubeVIS overlapping classes.
* openimages2det100.py => process openimages masks to a COCO format, unifying YoutubeVIS and COCO classes.
