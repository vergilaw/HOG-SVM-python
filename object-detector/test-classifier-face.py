#!/usr/bin/python
"""
Test the trained face detector on images
Uses sliding window and image pyramid for multi-scale detection
"""

from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
from nms import nms
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

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.

    Parameters:
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    Returns:
    * (x, y, im_window) tuple where:
      - x is the top-left x co-ordinate
      - y is the top-left y co-ordinate
      - im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    parser.add_argument('-d', '--downscale', help="Downscale ratio", default=1.25,
                       type=float)
    parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
                       action="store_true")
    args = vars(parser.parse_args())

    # Read the image
    im = imread(args["image"])
    downscale = args['downscale']
    visualize_det = args['visualize']

    # Load the classifier
    print "Loading face detector model from {}".format(model_path)
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0

    print "Detecting faces in the image..."
    print "Window size: {}".format(min_wdw_sz)
    print "Step size: {}".format(step_size)
    print "Downscale factor: {}".format(downscale)

    # Downscale the image and iterate
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break

        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue

            # Convert to grayscale for HOG
            if len(im_window.shape) == 3:
                im_window_gray = cv2.cvtColor((im_window * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
            else:
                im_window_gray = im_window

            # Calculate the HOG features
            fd = hog(im_window_gray, orientations, pixels_per_cell, cells_per_block,
                    visualise=visualize, transform_sqrt=transform_sqrt)

            # Reshape fd to match the expected input shape for prediction
            fd = fd.reshape(1, -1)

            pred = clf.predict(fd)
            if pred == 1:
                print "Detection:: Location -> ({}, {})".format(x, y)
                print "Scale ->  {} | Confidence Score {} \n".format(scale, clf.decision_function(fd))
                detections.append((x, y, clf.decision_function(fd)[0],
                                 int(min_wdw_sz[0] * (downscale ** scale)),
                                 int(min_wdw_sz[1] * (downscale ** scale))))
                cd.append(detections[-1])

            # If visualize is set to true, display the working of the sliding window
            if visualize_det:
                clone = im_scaled.copy()
                for x1, y1, _, _, _ in cd:
                    # Draw the detections at this scale
                    cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 + im_window.shape[0]),
                                (0, 0, 0), thickness=2)
                cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y + im_window.shape[0]),
                            (255, 255, 255), thickness=2)
                cv2.imshow("Sliding Window in Progress", clone)
                cv2.waitKey(30)

        # Move to the next scale
        scale += 1

    print "\nTotal detections before NMS: {}".format(len(detections))

    # Display the results before performing NMS
    clone = im.copy()
    if len(im.shape) == 2:
        clone = cv2.cvtColor((clone * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)
    elif im.max() <= 1.0:
        clone = (clone * 255).astype('uint8')

    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)
    cv2.imshow("Raw Detections before NMS", clone)
    cv2.waitKey()

    # Perform Non Maxima Suppression
    detections = nms(detections, threshold)
    print "Total detections after NMS: {}".format(len(detections))

    # Display the results after performing NMS
    clone = im.copy()
    if len(im.shape) == 2:
        clone = cv2.cvtColor((clone * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)
    elif im.max() <= 1.0:
        clone = (clone * 255).astype('uint8')

    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)
    cv2.imshow("Final Face Detections after applying NMS", clone)
    cv2.waitKey()
