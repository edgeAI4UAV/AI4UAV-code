import argparse
import os
import cv2
import sys
import numpy as np
import json

parser = argparse.ArgumentParser(description='Extract Target Person Coordinates in image')
parser.add_argument('--image_file_fullpath', help='full path to image file')
parser.add_argument('--txt_name_file', help='txt name file of target coordinates in image')

args = parser.parse_args()

image = cv2.imread(args.image_file_fullpath, 1)

# Select ROI
person_coordinates = cv2.selectROI("select the area", image)

x_pixel_coordinate = int(person_coordinates[0])
y_pixel_coordinate = int(person_coordinates[1])
bbox_width_coordinates = int(person_coordinates[2])
bbox_height_coordinates = int(person_coordinates[3])

# Crop image
cropped_image = image[int(person_coordinates[1]):int(person_coordinates[1] + person_coordinates[3]),
                int(person_coordinates[0]):int(person_coordinates[0] + person_coordinates[2])]

a = {'x_pixel_coordinate': x_pixel_coordinate, 'y_pixel_coordinate': y_pixel_coordinate,
     'bbox_width_coordinates': bbox_width_coordinates, 'bbox_height_coordinates': bbox_height_coordinates}

with open(args.txt_name_file, "w") as fp:
    json.dump(a, fp)
