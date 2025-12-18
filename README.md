# 🚗♻️ EcoWheels - AI-Based Smart Waste Sorting System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-green)](https://docs.ultralytics.com/)

An autonomous waste detection and sorting system using YOLOv11 on an RC car platform with a robotic arm. Detects and classifies waste in real-time for proper recycling.

## 🎯 Overview

EcoWheels addresses waste management challenges in densely populated areas by combining:
- **Real-time Detection**: YOLOv11-based waste identification
- **Autonomous Navigation**: RC car platform for waste collection
- **Smart Sorting**: Multi-class waste classification
- **Automated Pickup**: Robotic arm integration

## 📊 Dataset

We use the **[TACO Dataset](https://www.kaggle.com/datasets/kneroma/tacotrashdataset)** (Trash Annotations in Context):
- **1,500+ images** of real-world waste in diverse environments
- Captured in streets, beaches, parks, and natural settings
- Annotated and refined using Label Studio
- Classes: Plastic, metal, paper, glass, organic waste, and more

**Why TACO?**  
Real-world complexity with varied lighting, backgrounds, and angles - perfect for outdoor autonomous collection.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Ipuuuu/EcoWheels_Proj.git
cd ecowheels

# Install dependencies
pip install -r requirements.txt

# Download dataset
kaggle datasets download -d kneroma/tacotrashdataset -p data/raw

# Train model
yolo detect train data=data/taco_dataset/data.yaml model=yolo11s.pt epochs=100

# Test on webcam
yolo detect predict model=runs/detect/train/weights/best.pt source=0
```


## 💻 Usage

### Live Inference (Camera → YOLO → Browser)

EcoWheels runs real-time object detection using a **phone camera (DroidCam)** streamed to a **local FastAPI server**, where YOLOv11 performs inference. The annotated video stream is then viewed in any browser.

> ⚠️ **Note:** For demos with no internet connection, all devices must be on the **same local network (LAN)**.

---

### 1️⃣ Start DroidCam (Phone)

* Install **DroidCam** on your phone
* Connect your phone and PC to the **same Wi-Fi network**
* Start the camera stream
* Note the stream URL (example):

  ```
  http://192.168.5.117:4747/video
  ```

---

### 2️⃣ Run the FastAPI YOLO Server (PC)

```bash
cd src
python server.py
```

Or using uvicorn:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server will:

* Read frames from DroidCam
* Run YOLOv11 inference
* Stream annotated frames as MJPEG

---

### 3️⃣ View the Live Detection Stream

Open a browser on **any device on the same LAN** and navigate to:

```
http://<PC_LAN_IP>:8000/video
```

Example:

```
http://192.168.5.42:8000/video
```

You should see:

* Live camera feed
* Bounding boxes and class labels
* Real-time detection updates

---

### 🌐 Optional: Remote Viewing (Internet Required)

For remote access (development/demo only), you can expose the local server using **ngrok**:

```bash
ngrok http 8000
```

This generates a public URL such as:

```
https://abc123.ngrok-free.app/video
```

Anyone with the link can view the live detection stream while the server is running.

> ⚠️ Not recommended for offline demos or unstable networks.

---

### 🧪 Supported Input Modes

| Source                  | Status            |
| ----------------------- | ----------------- |
| Phone camera (DroidCam) | ✅ Recommended     |
| Local webcam            | ✅ Supported       |
| Image file              | ✅ Supported       |
| Offline LAN demo        | ✅ Fully supported |
| Cloud-only / serverless | ❌ Not supported   |

---

### 📝 Notes for Demos

* Reduce camera resolution (e.g. 640×480) for stability
* Limit FPS (5–10 FPS) for smoother streaming
* Prefer **local LAN setup** over internet tunneling
* Vercel frontend is optional and not required for offline demos



## 📈 Results

| Metric | Score |
|--------|-------|
| mAP@0.5 | 13.5% |
| Precision | 19.7% |
| Recall | 14.4% |
| Speed | ~40ms/image |

## 📁 Project Structure

```
EcoWheels_Proj/
├── data/                          # TACO dataset (YOLO format)
│   ├── images/                   # Raw images
│   ├── labels/                   # YOLO format annotations
│   ├── train/                    # Training split
│   ├── validation/               # Validation split
│   └── classes.txt              # Class definitions
├── models/                        # Trained model weights & notebooks
│   ├── Eco.ipynb                # Training & analysis notebook
│   ├── yolo11n.pt               # Nano model
│   ├── yolo11s.pt               # Small model
│   └── runs/                    # Training runs & results
├── yolo_taco/                     # YOLO-formatted dataset splits
│   ├── train/
│   ├── val/
│   ├── test/
│   └── data.yaml
├── yolo_taco_material_merged/     # Material-based class merged dataset
│   ├── train/
│   ├── val/
│   ├── test/
│   └── data.yaml
├── frontend/                      # React/Next.js demo interface
├── src/
│   └── server.py                # FastAPI server for live detection
├── Org_dataset/                   # Original dataset with annotations
├── runs/                          # YOLO training outputs
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 👥 Team

- **AI/ML Lead**: Model training and optimization
- **Data Engineer**: Dataset preparation and annotation
- **CV Specialist**: Real-time detection integration
- **Hardware Engineer**: RC car and robotic arm
- **Full-Stack Dev**: Demo interface and coordination

## 📚 References

- [TACO Dataset Paper](https://arxiv.org/abs/2003.06975) - Proença & Simões (2020)
- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [Label Studio](https://labelstud.io/)

## 📄 License

MIT License - See [LICENSE](https://github.com/Ipuuuu/EcoWheels_Proj/blob/main/LICENSE) for details.

---

**Building a cleaner future through AI and Robotics** 🌍✨
