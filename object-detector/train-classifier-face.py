#!/usr/bin/python
"""
Train SVM classifier for face detection using extracted HOG features
"""

from sklearn.svm import LinearSVC
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
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--posfeat", help="Path to the positive features directory",
                       default="../data/features/face/pos")
    parser.add_argument('-n', "--negfeat", help="Path to the negative features directory",
                       default="../data/features/face/neg")
    parser.add_argument('-c', "--classifier", help="Classifier to be used", default="LIN_SVM")
    args = vars(parser.parse_args())

    pos_feat_path = args["posfeat"]
    neg_feat_path = args["negfeat"]
    clf_type = args['classifier']

    fds = []
    labels = []

    # Load the positive features
    print "Loading positive features..."
    count = 0
    for feat_path in glob.glob(os.path.join(pos_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)
        count += 1
    print "Loaded {} positive features".format(count)

    # Load the negative features
    print "Loading negative features..."
    count = 0
    for feat_path in glob.glob(os.path.join(neg_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
        count += 1
    print "Loaded {} negative features".format(count)

    if clf_type == "LIN_SVM":
        print "\nTraining a Linear SVM Classifier for Face Detection..."
        print "Total training samples: {}".format(len(fds))
        clf = LinearSVC(random_state=42, max_iter=2000, verbose=1)
        clf.fit(fds, labels)

        # If model directory doesn't exist, create it
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])

        joblib.dump(clf, model_path)
        print "Classifier saved to {}".format(model_path)
        print "\nTraining completed successfully!"
        print "You can now test the face detector using test-classifier-face.py"
