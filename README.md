# Face Detector - HOG + SVM

Phát hiện khuôn mặt trong ảnh sử dụng HOG (Histogram of Oriented Gradients) features và Linear SVM classifier.

## Tính năng

- Phát hiện khuôn mặt trong ảnh với độ chính xác cao
- Sử dụng kỹ thuật sliding window và image pyramid để phát hiện đa tỷ lệ
- Pipeline tự động từ chuẩn bị dataset đến training và testing
- Dễ dàng cấu hình và tùy chỉnh tham số

## Cài đặt

### Yêu cầu hệ thống

- Python 2.7
- OpenCV
- scikit-learn
- scikit-image
- numpy

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Sử dụng nhanh

### Cách 1: Chạy pipeline tự động (Khuyến nghị)

```bash
cd bin
python main.py
```

Script này sẽ tự động:
1. Download dataset LFW (Labeled Faces in the Wild)
2. Chuẩn bị positive và negative samples
3. Trích xuất HOG features
4. Huấn luyện SVM classifier
5. Test trên ảnh mẫu

### Cách 2: Chạy từng bước thủ công

```bash
# Bước 1: Chuẩn bị dataset
cd bin
python prepare-dataset.py

# Bước 2: Trích xuất features
python ../object-detector/extract-features.py \
    -p ../data/dataset/faces/pos \
    -n ../data/dataset/faces/neg

# Bước 3: Huấn luyện classifier
python ../object-detector/train-classifier.py \
    -p ../data/features/face/pos \
    -n ../data/features/face/neg

# Bước 4: Test trên ảnh của bạn
python ../object-detector/test-classifier.py -i /path/to/your/image.jpg
```

## Sử dụng trên PyCharm

### 1. Mở project trong PyCharm

1. Mở PyCharm
2. Chọn **File** > **Open**
3. Chọn thư mục `HOG-SVM-python`
4. Click **OK**

### 2. Cấu hình Python Interpreter

1. Chọn **File** > **Settings** (hoặc **PyCharm** > **Preferences** trên macOS)
2. Vào **Project: HOG-SVM-python** > **Python Interpreter**
3. Click biểu tượng **⚙️** > **Add**
4. Chọn **Virtualenv Environment** > **New environment**
5. Click **OK**
6. Chờ PyCharm tạo virtual environment

### 3. Cài đặt Dependencies trong PyCharm

**Cách 1: Từ requirements.txt**
1. Mở file `requirements.txt`
2. PyCharm sẽ hiện thông báo "Package requirements are not satisfied"
3. Click **Install requirements**

**Cách 2: Thủ công**
1. Mở Terminal trong PyCharm (**View** > **Tool Windows** > **Terminal**)
2. Chạy:
```bash
pip install -r requirements.txt
```

### 4. Chạy Main Script trong PyCharm

1. Mở file `bin/main.py`
2. Click chuột phải vào file
3. Chọn **Run 'main'**

Hoặc:
1. Mở `main.py`
2. Nhấn **Shift + F10** (Windows/Linux) hoặc **Control + R** (macOS)

### 5. Chạy Test Classifier trong PyCharm

1. Mở file `object-detector/test-classifier.py`
2. Tạo Run Configuration:
   - Click **Run** > **Edit Configurations**
   - Click **+** > **Python**
   - Name: `Test Face Detector`
   - Script path: Chọn `object-detector/test-classifier.py`
   - Parameters: `-i /path/to/image.jpg -d 1.25`
   - Working directory: `object-detector`
   - Click **OK**
3. Click **Run** > **Run 'Test Face Detector'**

### 6. Debug trong PyCharm

1. Đặt breakpoint bằng cách click vào lề trái của dòng code
2. Click chuột phải vào file
3. Chọn **Debug 'filename'**
4. Sử dụng Debug panel để:
   - Step Over (F8)
   - Step Into (F7)
   - Resume Program (F9)
   - Xem biến trong Variables panel

### 7. Tổ chức Project trong PyCharm

Cấu trúc project:
```
HOG-SVM-python/
├── bin/                      # Scripts chính
│   ├── main.py              # Pipeline tự động
│   └── prepare-dataset.py   # Chuẩn bị dataset
├── object-detector/         # Module phát hiện
│   ├── extract-features.py  # Trích xuất HOG features
│   ├── train-classifier.py  # Huấn luyện SVM
│   ├── test-classifier.py   # Test detector
│   └── nms.py              # Non-maximum suppression
├── data/
│   ├── config/
│   │   └── config.cfg      # Cấu hình HOG & SVM
│   ├── dataset/            # Dataset ảnh
│   ├── features/           # HOG features
│   └── models/             # Trained models
└── requirements.txt        # Python dependencies
```

## Cấu hình

File cấu hình: `data/config/config.cfg`

```ini
[hog]
min_wdw_sz: [64, 64]        # Kích thước cửa sổ
step_size: [8, 8]           # Bước trượt
orientations: 9             # Số hướng gradient
pixels_per_cell: [8, 8]     # Pixel mỗi cell
cells_per_block: [2, 2]     # Cell mỗi block
visualize: False            # Visualize HOG
transform_sqrt: True        # Chuẩn hóa sqrt

[nms]
threshold: 0.3              # Ngưỡng NMS

[paths]
pos_feat_ph: ../data/features/face/pos
neg_feat_ph: ../data/features/face/neg
model_path: ../data/models/face_svm.model
```

## Tùy chỉnh và Mở rộng

### Thay đổi dataset

Để sử dụng dataset khác:
1. Chuẩn bị ảnh positive (có khuôn mặt) 64x64 pixels
2. Chuẩn bị ảnh negative (không có khuôn mặt) 64x64 pixels
3. Đặt vào `data/dataset/faces/pos/` và `data/dataset/faces/neg/`
4. Chạy feature extraction và training

### Tùy chỉnh tham số HOG

Chỉnh sửa `data/config/config.cfg`:
- Tăng `orientations` để capture chi tiết hơn
- Giảm `step_size` để tăng độ chính xác (nhưng chậm hơn)
- Thay đổi `min_wdw_sz` nếu khuôn mặt có kích thước khác

### Tối ưu hiệu năng

- Tăng `step_size` để tăng tốc độ
- Tăng `downscale` factor khi test
- Giảm số lượng orientations
- Sử dụng Hard Negative Mining

## Các Module

### bin/prepare-dataset.py
Download và chuẩn bị dataset LFW cho face detection.

### object-detector/extract-features.py
Trích xuất HOG features từ ảnh training.

Parameters:
- `-p, --pospath`: Đường dẫn đến ảnh positive
- `-n, --negpath`: Đường dẫn đến ảnh negative
- `-d, --descriptor`: Loại descriptor (mặc định: HOG)

### object-detector/train-classifier.py
Huấn luyện Linear SVM classifier.

Parameters:
- `-p, --posfeat`: Đường dẫn đến positive features
- `-n, --negfeat`: Đường dẫn đến negative features
- `-c, --classifier`: Loại classifier (mặc định: LIN_SVM)

### object-detector/test-classifier.py
Test face detector trên ảnh.

Parameters:
- `-i, --image`: Đường dẫn đến ảnh test (required)
- `-d, --downscale`: Tỷ lệ downscale cho image pyramid (default: 1.25)
- `-v, --visualize`: Hiển thị quá trình sliding window

Example:
```bash
python test-classifier.py -i photo.jpg -d 1.3 -v
```

### object-detector/nms.py
Non-Maximum Suppression để loại bỏ các detection trùng lặp.

## Kết quả mẫu

Sau khi chạy detection, bạn sẽ thấy:
1. **Raw Detections before NMS**: Tất cả các detection trước khi lọc
2. **Final Detections after NMS**: Detection cuối cùng sau khi loại bỏ trùng lặp

## Khắc phục sự cố

### Lỗi import cv2
```bash
pip install opencv-python
```

### Lỗi sklearn.externals
Nếu dùng scikit-learn >= 0.23:
```bash
pip install scikit-learn==0.22.2
```

### Lỗi không tìm thấy model
Đảm bảo đã chạy training:
```bash
python train-classifier.py -p ../data/features/face/pos -n ../data/features/face/neg
```

### Phát hiện không chính xác
- Thêm nhiều ảnh training hơn
- Cải thiện chất lượng negative samples
- Điều chỉnh tham số trong config.cfg
- Sử dụng Hard Negative Mining

## Tài liệu tham khảo

1. [Histogram of Oriented Gradients and Object Detection](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)
2. [Image Pyramids with Python and OpenCV](http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/)
3. [Sliding Windows for Object Detection](http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
4. [Non-Maximum Suppression for Object Detection](http://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/)

## License

MIT License - xem file LICENSE để biết thêm chi tiết.

## Credits

Based on the original HOG-SVM object detector, adapted specifically for face detection.
