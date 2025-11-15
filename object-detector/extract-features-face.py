#!/usr/bin/python
"""
Extract HOG features from face images for training
"""

from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import argparse as ap
import glob
import os
import ConfigParser as cp
import json

# Load face detection config
config = cp.RawConfigParser()
config.read('../data/config/config-face.cfg')

min_wdw_sz = tuple(json.loads(config.get("hog","min_wdw_sz")))
step_size = tuple(json.loads(config.get("hog", "step_size")))
orientations = config.getint("hog", "orientations")
pixels_per_cell = tuple(json.loads(config.get("hog", "pixels_per_cell")))
cells_per_block = tuple(json.loads(config.get("hog", "cells_per_block")))
visualize = config.getboolean("hog", "visualize")
transform_sqrt = config.getboolean("hog", "transform_sqrt")
pos_feat_ph = config.get("paths", "pos_feat_ph")
neg_feat_ph = config.get("paths", "neg_feat_ph")
model_path = config.get("paths", "model_path")
threshold = config.getfloat("nms", "threshold")

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive face images",
            default="../data/dataset/faces/pos")
    parser.add_argument('-n', "--negpath", help="Path to negative images",
            default="../data/dataset/faces/neg")
    parser.add_argument('-d', "--descriptor", help="Descriptor to be used -- HOG",
            default="HOG")
    args = vars(parser.parse_args())

    pos_im_path = args["pospath"]
    neg_im_path = args["negpath"]
    des_type = args["descriptor"]

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)

    print "Calculating the descriptors for the positive face samples and saving them"
    count = 0
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        try:
            im = imread(im_path, as_grey=True)
            if des_type == "HOG":
                fd = hog(im, orientations, pixels_per_cell, cells_per_block,
                        visualise=visualize, transform_sqrt=transform_sqrt)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(pos_feat_ph, fd_name)
            joblib.dump(fd, fd_path)
            count += 1
            if count % 100 == 0:
                print "Processed {} positive samples...".format(count)
        except Exception as e:
            print "Error processing {}: {}".format(im_path, str(e))
            continue
    print "Positive features saved in {}".format(pos_feat_ph)
    print "Total positive samples: {}".format(count)

    print "\nCalculating the descriptors for the negative samples and saving them"
    count = 0
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        try:
            im = imread(im_path, as_grey=True)
            if des_type == "HOG":
                fd = hog(im, orientations, pixels_per_cell, cells_per_block,
                        visualise=visualize, transform_sqrt=transform_sqrt)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(neg_feat_ph, fd_name)
            joblib.dump(fd, fd_path)
            count += 1
            if count % 100 == 0:
                print "Processed {} negative samples...".format(count)
        except Exception as e:
            print "Error processing {}: {}".format(im_path, str(e))
            continue
    print "Negative features saved in {}".format(neg_feat_ph)
    print "Total negative samples: {}".format(count)

    print "\nCompleted calculating features from training images"
