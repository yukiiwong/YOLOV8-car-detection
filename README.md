# YOLOv8 + DeepSORT for Object Detection and Multi-Object Tracking

This repository provides an end-to-end pipeline combining **Ultralytics YOLOv8** for object detection and **DeepSORT** for multi-object tracking. The system takes a video as input, performs per-frame detection, assigns consistent track IDs across frames, and exports both visualization and structured tracking results (`output.csv`).




## Environment & Dependencies

### **Tested Environment**
| Component | Version |
|----------|---------|
| OS | Windows 10 / 11 |
| Python | 3.8.18 (Anaconda) |
| PyTorch | 2.2.2 + CUDA |
| OpenCV | 4.8.0.76 |
| Ultralytics YOLOv8 | 8.0.180 |

---

## Step-by-Step Installation (Conda)

### Create Conda Environment**
```bash
conda create -n yolov8 python=3.8 -y
````

### Activate the Environment**

```bash
conda activate yolov8
```

### Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation**

```bash
python - <<EOF
import torch, ultralytics, cv2
print("Torch:", torch.__version__)
print("YOLOv8:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```

If no errors occur, your environment is ready.

---

## Quick Start

### **Run Detection + Tracking**

```bash
python predict.py model=yolov8l.pt source="clip_1.mp4" show=True
```

### **Run without GUI display**

```bash
python predict.py source="clip_1.mp4" show=False
```

### **Output**

* Processed video stream (optional visualization)
* `output.csv` (tracking results with bounding boxes and identity numbers)

Example CSV structure:

| frame | x_min | y_min | x_max | y_max | class_id | class_name | identity |
| ----- | ----- | ----- | ----- | ----- | -------- | ---------- | -------- |

---

## YOLOv8 Model Variants

Different YOLOv8 model sizes can be selected depending on speed vs accuracy needs:

| Model                   | Speed     | Accuracy        | Recommended Use         |
| ----------------------- | --------- | --------------- | ----------------------- |
| **yolov8n**             | Fastest | Lowest         | Edge devices, real-time |
| **yolov8l** *(default)* | Slow      | **High accuracy** | Research / offline      |
| **yolov8x**             | Slowest | Best accuracy | High-end GPU            |

#### Example Switching Models

```bash
python predict.py model=yolov8n.pt source="clip_1.mp4"
python predict.py model=yolov8x.pt source="clip_1.mp4"
```

---

## 🏷Using Custom Trained Weights

To use your own trained model, simply replace `model` with your `.pt` file:

```bash
python predict.py model="runs/detect/train/weights/best.pt" source="test.mp4"
```

Make sure:

* Class labels match your dataset
* DeepSORT appearance features may need adjustments for very different objects

---

## Project Structure

```text
.
├── predict.py              # Main script (YOLOv8 + DeepSORT + CSV export)
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
└── deep_sort_pytorch/      # DeepSORT implementation
    ├── deep_sort.py
    ├── utils/
    │   └── parser.py
    └── configs/
        └── deep_sort.yaml
```

---


## Acknowledgements

* **Ultralytics YOLOv8** — [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* **DeepSORT Pytorch** — [https://github.com/ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)
* **Muhammad Moin** — [https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking)
This project is intended for academic and research purposes.

---

## 📬 Contact

If you have questions or suggestions, please open an issue or contact yukai@kaist.ac.kr.

