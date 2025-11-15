#!/usr/bin/python
"""
Script to download and prepare face detection dataset.
This script will:
1. Download a face dataset (LFW - Labeled Faces in the Wild)
2. Prepare positive samples (cropped faces)
3. Prepare negative samples (non-face images)
"""

import os
import sys
import urllib
import tarfile
import cv2
import numpy as np
from glob import glob

def create_directories():
    """Create necessary directories for face detection"""
    dirs = [
        "../data/dataset/faces",
        "../data/dataset/faces/pos",
        "../data/dataset/faces/neg",
        "../data/features/face",
        "../data/features/face/pos",
        "../data/features/face/neg"
    ]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print("Created directory: {}".format(dir_path))

def download_lfw_dataset():
    """
    Download LFW (Labeled Faces in the Wild) dataset
    This is a simplified version - you can extend it
    """
    dataset_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    dataset_path = "../data/dataset/faces/lfw.tgz"

    if not os.path.exists(dataset_path):
        print("Downloading LFW dataset...")
        print("This may take a while depending on your internet connection...")
        try:
            urllib.urlretrieve(dataset_url, dataset_path)
            print("Download completed!")
        except Exception as e:
            print("Error downloading dataset: {}".format(str(e)))
            return False
    else:
        print("Dataset already exists at {}".format(dataset_path))

    # Extract dataset
    if not os.path.exists("../data/dataset/faces/lfw"):
        print("Extracting dataset...")
        try:
            tar = tarfile.open(dataset_path)
            tar.extractall("../data/dataset/faces/")
            tar.close()
            print("Extraction completed!")
        except Exception as e:
            print("Error extracting dataset: {}".format(str(e)))
            return False

    return True

def prepare_positive_samples():
    """
    Prepare positive samples (face images)
    Resize all face images to 64x64
    """
    print("\nPreparing positive samples...")
    lfw_path = "../data/dataset/faces/lfw"
    pos_path = "../data/dataset/faces/pos"

    if not os.path.exists(lfw_path):
        print("LFW dataset not found. Please run download first.")
        return False

    count = 0
    target_size = (64, 64)

    # Get all image files from LFW dataset
    for person_dir in os.listdir(lfw_path):
        person_path = os.path.join(lfw_path, person_dir)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path)

                    if img is not None:
                        # Resize to 64x64
                        img_resized = cv2.resize(img, target_size)

                        # Save to pos directory
                        output_name = "face_{:05d}.jpg".format(count)
                        output_path = os.path.join(pos_path, output_name)
                        cv2.imwrite(output_path, img_resized)
                        count += 1

                        if count % 100 == 0:
                            print("Processed {} positive samples...".format(count))

                        # Limit to 1000 positive samples for training
                        if count >= 1000:
                            break
            if count >= 1000:
                break

    print("Total positive samples prepared: {}".format(count))
    return True

def prepare_negative_samples():
    """
    Prepare negative samples (non-face images)
    You can use random crops from other images or download a non-face dataset
    """
    print("\nPreparing negative samples...")
    print("For negative samples, you have several options:")
    print("1. Download a background/scene dataset")
    print("2. Use random crops from existing images")
    print("3. Use ImageNet or COCO dataset samples without faces")
    print("\nFor this example, we'll create synthetic negative samples.")
    print("You should replace this with real negative images for better results.")

    neg_path = "../data/dataset/faces/neg"
    count = 0
    target_size = (64, 64)

    # Create some simple synthetic negative samples (random noise, gradients, etc.)
    for i in range(500):
        # Random noise
        if i < 200:
            img = np.random.randint(0, 255, (target_size[0], target_size[1], 3), dtype=np.uint8)
        # Gradient patterns
        else:
            img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            for y in range(target_size[0]):
                img[y, :] = [y*4, y*4, y*4]

        output_name = "neg_{:05d}.jpg".format(i)
        output_path = os.path.join(neg_path, output_name)
        cv2.imwrite(output_path, img)
        count += 1

    print("Total negative samples prepared: {}".format(count))
    print("\nNOTE: These are synthetic negative samples.")
    print("For better results, please add real non-face images to: {}".format(neg_path))
    return True

def main():
    print("=" * 60)
    print("Face Detection Dataset Preparation")
    print("=" * 60)

    # Create directories
    create_directories()

    # Download dataset
    if not download_lfw_dataset():
        print("Failed to download dataset. Exiting.")
        return

    # Prepare positive samples
    if not prepare_positive_samples():
        print("Failed to prepare positive samples. Exiting.")
        return

    # Prepare negative samples
    if not prepare_negative_samples():
        print("Failed to prepare negative samples. Exiting.")
        return

    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the prepared samples in:")
    print("   - Positive: ../data/dataset/faces/pos")
    print("   - Negative: ../data/dataset/faces/neg")
    print("\n2. Add more negative samples (recommended)")
    print("\n3. Run feature extraction:")
    print("   python ../object-detector/extract-features-face.py")
    print("\n4. Train the classifier:")
    print("   python ../object-detector/train-classifier-face.py")

if __name__ == "__main__":
    main()
