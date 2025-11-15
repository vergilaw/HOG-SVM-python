#!/usr/bin/python
"""
Complete Face Detection Pipeline
This script will guide you through the complete face detection workflow:
1. Download and prepare face dataset
2. Extract HOG features from face images
3. Train SVM classifier
4. Test on sample images
"""

import os
import sys

def print_banner(text):
    """Print a formatted banner"""
    print "\n" + "=" * 70
    print text
    print "=" * 70 + "\n"

def print_step(step_num, text):
    """Print a formatted step"""
    print "\n--- Step {}: {} ---\n".format(step_num, text)

def main():
    print_banner("FACE DETECTION SYSTEM - Complete Pipeline")

    print "This script will set up and test a complete face detection system."
    print "The system uses HOG (Histogram of Oriented Gradients) features"
    print "combined with a Linear SVM classifier."

    # Check if dataset exists
    dataset_exists = os.path.exists("../data/dataset/faces/pos") and \
                    os.path.exists("../data/dataset/faces/neg")

    features_exist = os.path.exists("../data/features/face/pos") and \
                    os.path.exists("../data/features/face/neg")

    model_exists = os.path.exists("../data/models/face_svm.model")

    # Step 1: Prepare dataset
    if not dataset_exists:
        print_step(1, "Preparing Face Dataset")
        print "Dataset not found. Running dataset preparation script..."
        print "This will download the LFW (Labeled Faces in the Wild) dataset."
        print "This may take several minutes depending on your internet connection."

        response = raw_input("\nDo you want to proceed? (y/n): ")
        if response.lower() == 'y':
            os.system("python prepare-face-dataset.py")
        else:
            print "Skipping dataset preparation."
            print "You can run it manually later with: python prepare-face-dataset.py"
    else:
        print_step(1, "Dataset Check")
        print "Dataset already exists at ../data/dataset/faces/"

    # Step 2: Extract features
    if not features_exist:
        print_step(2, "Extracting HOG Features")
        print "Extracting features from face images..."

        pos_path = "../data/dataset/faces/pos"
        neg_path = "../data/dataset/faces/neg"

        if os.path.exists(pos_path) and os.path.exists(neg_path):
            cmd = "python ../object-detector/extract-features-face.py -p {} -n {}".format(
                pos_path, neg_path)
            os.system(cmd)
        else:
            print "ERROR: Dataset not found. Please run Step 1 first."
            return
    else:
        print_step(2, "Feature Check")
        print "Features already extracted at ../data/features/face/"

    # Step 3: Train classifier
    if not model_exists:
        print_step(3, "Training SVM Classifier")
        print "Training the Linear SVM classifier for face detection..."

        pos_feat_path = "../data/features/face/pos"
        neg_feat_path = "../data/features/face/neg"

        if os.path.exists(pos_feat_path) and os.path.exists(neg_feat_path):
            cmd = "python ../object-detector/train-classifier-face.py -p {} -n {}".format(
                pos_feat_path, neg_feat_path)
            os.system(cmd)
        else:
            print "ERROR: Features not found. Please run Step 2 first."
            return
    else:
        print_step(3, "Model Check")
        print "Trained model already exists at ../data/models/face_svm.model"

    # Step 4: Test on sample image
    print_step(4, "Testing Face Detector")

    # Check if we have a test image
    test_image = None

    # Look for test images in the dataset
    test_dirs = [
        "../data/dataset/faces/lfw",
        "../data/dataset/faces/pos"
    ]

    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            # Find first image in directory
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                        test_image = os.path.join(root, file)
                        break
                if test_image:
                    break
        if test_image:
            break

    if test_image:
        print "Found test image: {}".format(test_image)
        print "\nRunning face detection..."
        print "(Close the image windows to continue)"

        cmd = "python ../object-detector/test-classifier-face.py -i {} -d 1.25".format(
            test_image)
        os.system(cmd)
    else:
        print "No test image found."
        print "You can test the detector manually with:"
        print "  python ../object-detector/test-classifier-face.py -i <image_path>"

    # Final summary
    print_banner("Face Detection Setup Complete!")

    print "Your face detection system is ready to use!"
    print "\nTo detect faces in your own images, run:"
    print "  python ../object-detector/test-classifier-face.py -i <image_path>"
    print "\nOptional parameters:"
    print "  -d <downscale>   : Downscale factor for image pyramid (default: 1.25)"
    print "  -v              : Visualize sliding window process"
    print "\nExample:"
    print "  python ../object-detector/test-classifier-face.py -i my_photo.jpg -d 1.3"

    print "\nConfiguration file: ../data/config/config-face.cfg"
    print "Model file: ../data/models/face_svm.model"

    print "\n" + "=" * 70

if __name__ == "__main__":
    main()
