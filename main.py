#utility for converting annotation mask to COCO json
#Author: Usman Zahidi (Lincoln Agri-Robotics)

from os import listdir
import cv2, argparse
import numpy as np
from pycococreatortools import pycococreatortools
import datetime
import json
from pycococreatortools.GenericMask import GenericMask
from skimage import measure

parser = argparse.ArgumentParser(description="Mask to COCO json converter")
parser.add_argument("-i", "--image_dir", default='', type=str, metavar="PATH", help="path to image folder")
parser.add_argument("-m", "--annotation_masks_dir", default='', type=str, metavar="PATH", help="annotation mask folder")
parser.add_argument("-f", "--output_json_file", default='', type=str, metavar="PATH", help="annotation mask folder")


def write_json(common_list, image_dir,ann_dir, output_json_file,class_data):
    image_list=list()
    category_list = list()
    ann_list = list()

    annotation_id = 1

    dict_info    = pycococreatortools.create_info()
    dict_license = pycococreatortools.create_license_info()

    classes   = class_data.keys()
    for class_id in classes:
        category_list.append(pycococreatortools.create_categories_info(int(class_id), class_data[class_id][0][0]))

    dict_category = {"categories": category_list}
    image_id=0
    for image in common_list:
        col_image = cv2.imread(image_dir + image)
        if col_image is None:
            continue
        else:
            image_size = [col_image.shape[0], col_image.shape[1]]
            image_list.append(pycococreatortools.create_image_info(image_id, image, image_size,
                                                                   datetime.datetime.utcnow().isoformat(' ')))
            annotation_image=cv2.imread(ann_dir +  image)
            masks_class = list()
            masks_image = list()
            masks=list()

            for class_id in class_data:
                r,g,b=class_data[class_id][1]
                class_mask = np.bitwise_and(annotation_image[:, :, 1] == g, annotation_image[:, :, 2] == r)
                class_mask = np.bitwise_and(annotation_image[:, :, 0] == b, class_mask == True)
                masks_class.append(class_mask)
                masks_image.append(class_mask)

                mask=(1*masks_class[int(class_id)-1]).astype(np.uint8)

                all_labels = measure.label(mask,None,False,2)
                components=np.unique(all_labels).tolist()
                components = [i for i in components if i != 0]
                for comp in components:
                    mask=(1*(all_labels==comp)).astype(np.uint8)
                    masks.append(GenericMask(mask, col_image.shape[0], col_image.shape[1]))

                for segment in masks:
                    category_info = {'id': int(class_id), 'is_crowd': '0'}
                    annotation_info,area,bbox_ar = pycococreatortools.create_annotation_info(
                        annotation_id, image_id, category_info, segment.mask, 0)
                    ann_list.append(annotation_info)
                    annotation_id += 1
                masks.clear()
            image_id += 1
            print("processed images: " + str(image_id))
    dict_images = {"images": image_list}
    dict_annotations = {"annotations": ann_list}

    json_output_dict = {**dict_info, **dict_license, **dict_images, **dict_annotations, **dict_category}


    with open(output_json_file.format(output_json_file), 'w') as output_json_file:
        json.dump(json_output_dict, output_json_file, skipkeys=False, ensure_ascii=True, check_circular=True,
                  allow_nan=True, cls=None, indent=2, separators=None, default=None, sort_keys=False)

def main():

    args = parser.parse_args()
    print(args.image_dir)
    print(args.annotation_masks_dir)
    print(args.output_json_file)

    assert (not (not args.image_dir or not args.annotation_masks_dir or not args.output_json_file)), "invalid parameters provided"

    image_dir = args.image_dir if args.image_dir[-1] == '/' else args.image_dir + '/'
    ann_dir = args.annotation_masks_dir if args.annotation_masks_dir[-1] == '/' else args.annotation_masks_dir + '/'
    output_json_file = args.output_json_file

    image_list = listdir(image_dir)
    ann_list = listdir(ann_dir)
    common_list=list()

    with open('./data/class_definition.json') as json_file:
        class_data = json.load(json_file)

    for image in image_list:
        image_ann=image
        if image_ann in ann_list:
            common_list.append(image)
    write_json(common_list, image_dir, ann_dir, output_json_file,class_data)

if __name__ == '__main__':
    main()
