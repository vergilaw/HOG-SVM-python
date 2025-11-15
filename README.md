# object-detector
Object Detector using HOG as descriptor and Linear SVM as classifier. | [Video](https://www.youtube.com/watch?v=SPXocFBjr70)

**NEW:** Now supports both **Car Detection** (original) and **Face Detection**!

## Quick Start

### Option 1: Car Detection (Original)

Run the car detector with the original functionality.

### Option 2: Face Detection (New!)

Detect faces in images using the same HOG+SVM approach.

```shell
cd bin
python test-face-detector.py
```

This will:
1. Download the LFW (Labeled Faces in the Wild) dataset
2. Extract HOG features from face images
3. Train a Linear SVM classifier
4. Test the face detector on sample images

## Run the code

I have created a single python script that can be used to test the code. To test the code, run the lines below in your terminal.

```shell
git clone https://github.com/bikz05/object-detector.git
cd object-detector/bin
test-object-detector
```

_The `test-object-detector` will download the [UIUC Image Database for Car Detection](https://cogcomp.cs.illinois.edu/Data/Car/) and train a classifier to detect cars in an image. The SVM model files will be stored in `data/models`, so that they can be resused later on._

### Configuration File

All the configurations are in the `data/config/config.cfg` configuration files. You can change it as per your need. Here is what the default configuration file looks like (which I have set for Car Detector)-

```bash
[hog]
min_wdw_sz: [100, 40]
step_size: [10, 10]
orientations: 9
pixels_per_cell: [8, 8]
cells_per_block: [3, 3]
visualize: False
normalize: True

[nms]
threshold: .3

[paths]
pos_feat_ph: ../data/features/pos
neg_feat_ph: ../data/features/neg
model_path: ../data/models/svm.model
```

### About the modules

* `extract-features.py` -- This module is used to extract HOG features of the training images.
* `train-classifier.py` -- This module is used to train the classifier.
* `nms.py` -- This module performs Non Maxima Suppression.
* `test-classifier.py` -- This module is used to test the classifier using a test image.
* `config.py` -- Imports the configuration variables from `config.cfg`.

## Some of the results

#### Test Image 1
_Detections before NMS_

![Image 1](data/images/test-im-1.png)

_Detections after NMS_

![](data/images/test-im-1-nms.png)
#### Test Image 2
_Detections before NMS_

![](data/images/test-im-2.png)

_Detections after NMS_

![](data/images/test-im-2-nms.png)
#### Test Image 3
_Detections before NMS_

![](data/images/test-im-3.png)

_Detections after NMS_

![](data/images/test-im-3-nms.png)
#### Test Image 4
_Detections before NMS_

![](data/images/test-im-4.png)

_Detections after NMS_

![](data/images/test-im-4-nms.png)

## Face Detection Module

This project now includes a complete face detection pipeline using the same HOG+SVM architecture.

### Face Detection Configuration

The face detection configuration is in `data/config/config-face.cfg`:

```bash
[hog]
min_wdw_sz: [64, 64]
step_size: [8, 8]
orientations: 9
pixels_per_cell: [8, 8]
cells_per_block: [2, 2]
visualize: False
transform_sqrt: True

[nms]
threshold: 0.3

[paths]
pos_feat_ph: ../data/features/face/pos
neg_feat_ph: ../data/features/face/neg
model_path: ../data/models/face_svm.model
```

### Face Detection Modules

* `prepare-face-dataset.py` -- Downloads and prepares the LFW face dataset
* `extract-features-face.py` -- Extracts HOG features from face images
* `train-classifier-face.py` -- Trains the SVM classifier for faces
* `test-classifier-face.py` -- Tests the face detector on images
* `test-face-detector.py` -- Complete pipeline script (recommended)

### Manual Face Detection Workflow

If you want to run each step manually:

```shell
# Step 1: Prepare dataset
cd bin
python prepare-face-dataset.py

# Step 2: Extract features
python ../object-detector/extract-features-face.py

# Step 3: Train classifier
python ../object-detector/train-classifier-face.py

# Step 4: Test on an image
python ../object-detector/test-classifier-face.py -i /path/to/image.jpg
```

### Face Detection Usage

After training, you can detect faces in any image:

```shell
python ../object-detector/test-classifier-face.py -i your_image.jpg -d 1.25
```

Parameters:
- `-i` : Path to input image (required)
- `-d` : Downscale factor for image pyramid (default: 1.25)
- `-v` : Visualize the sliding window process

## TODO

Here is list of tasks that I am planning to implement in the future -

* Optimize code to use more `numpy` vectorized codes.
* Faster NMS code.
* Add bootstrapping (Hard Negative Mining) code.
* Improve face detection with better negative samples
* Add support for more object types


## Useful tutorials

1. [Histogram of Oriented Gradients and Object Detection](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)
2. [Image Pyramids with Python and OpenCV](http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/)
3. [Sliding Windows for Object Detection with Python and OpenCV](http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
4. [Non-Maximum Suppression for Object Detection in Python](http://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/)
5. [(Faster) Non-Maximum Suppression in Python](http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)
6. [Texture Matching using Local Binary Patterns (LBP), OpenCV, scikit-learn and Python](http://hanzratech.in/2015/05/30/local-binary-patterns.html)
7. [Detección de objetos Course by Coursera](https://www.coursera.org/course/deteccionobjetos)
