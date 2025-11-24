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
git clone https://github.com/your-team/ecowheels.git
cd ecowheels

# Install dependencies
pip install -r requirements.txt

# Download dataset
kaggle datasets download -d kneroma/tacotrashdataset -p data/raw

# Train model
yolo detect train data=data/taco_dataset/data.yaml model=yolo11n.pt epochs=50

# Test on webcam
yolo detect predict model=runs/detect/train/weights/best.pt source=0
```

## 💻 Usage

### Inference
```bash
# Webcam
python src/detection/detector.py --source 0 --weights models/best.pt

# Image
python src/detection/detector.py --source path/to/image.jpg --weights models/best.pt

# Demo Interface
cd frontend && npm start
```

## 📈 Results

| Metric | Score |
|--------|-------|
| mAP@0.5 | 79.1% |
| Precision | 74.1% |
| Recall | 77.2% |
| Speed | ~40ms/image |

## 📁 Project Structure

```
ecowheels/
├── data/                  # TACO dataset (YOLO format)
├── models/                # Trained weights
├── src/
│   ├── detection/        # Inference code
│   ├── hardware/         # RC car & arm control
│   └── models/           # Training scripts
├── frontend/             # React demo
└── requirements.txt
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

MIT License - See [LICENSE](LICENSE) for details.

---

**Building a cleaner future through AI and Robotics** 🌍✨
