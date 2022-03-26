import csv

images_to_skip = []

with open("train-images-boxable-with-rotation.csv", newline='') as csvfile:
    table_reader = csv.reader(csvfile, delimiter=',')

    skipped_first = False
    for row in table_reader:

        if not skipped_first:
            skipped_first = True
            continue


        """
        ['ImageID', 'Subset', 'OriginalURL', 'OriginalLandingURL', 'License',
        'AuthorProfileURL', 'Author', 'Title', 'OriginalSize', 'OriginalMD5',
        'Thumbnail300KURL', 'Rotation']
        """
        rotation = row[-1]
        image_id = row[0]

        if not rotation:
            print(rotation)
            images_to_skip.append(image_id)
        else:
            rotation = int(float(rotation))
            if rotation != 0:
                print(rotation)
                images_to_skip.append(image_id)

    print(len(images_to_skip))
    images_to_skip = sorted(images_to_skip)
    with open("images_to_skip.txt", "w") as f:
        for image_id in images_to_skip[:-2]:
            f.write(f"{image_id}\n")
        f.write(image_id)
