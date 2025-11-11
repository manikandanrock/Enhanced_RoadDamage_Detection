# Enhanced_RoadDamage_Detection
Deep learning–based road damage detection and segmentation using YOLOv8, ResNet, and an enhanced EfficientNet-B4 + U-Net hybrid model for smart-city infrastructure monitoring.
# 🛣️ Comparative Analysis of Deep Learning Models for Road Damage Detection and Segmentation

### Models Compared
**YOLOv8 | ResNet | Enhanced EfficientNet-B4 + U-Net**

---

## 📘 Overview
Automated detection and segmentation of road surface damage (potholes, cracks, and depressions) are critical for smart-city maintenance and road safety.  
This project presents a **comparative study** of three deep-learning models — ResNet (segmentation baseline), YOLOv8 (detection baseline), and an **Enhanced EfficientNet-B4 + U-Net** (proposed hybrid model).  

The proposed model integrates **channel-attention skip connections** and **deep supervision** to achieve superior segmentation accuracy and faster convergence.

---

## 🎯 Objectives
- Evaluate and compare the performance of **ResNet**, **YOLOv8**, and **Enhanced EfficientNet-B4 + U-Net** on a unified pothole dataset.  
- Design an improved segmentation model offering **higher accuracy**, **precise boundaries**, and **faster convergence**.  
- Enable **automated road-quality assessment** for municipal and smart-city systems.

---

## 🧠 Methodology

### Dataset
- **Source:** Farzad Nekouei’s *Pothole Image Segmentation Dataset* (Kaggle)  
- **Samples:** 780 annotated images  
- **Split:** 720 train / 60 validation  
- **Conditions:** varied lighting, wet/dry surfaces, damage size variation  
- **Resolution:** resized to 512×512 pixels  

### Preprocessing
- Normalization using ImageNet mean & std  
- Data augmentation:
  - Random flips and rotations  
  - Brightness/contrast changes  
  - Gaussian blur  
  - Random scaling (0.8 – 1.2×)

### Models Evaluated
| Model | Description | Key Points |
|--------|--------------|------------|
| **ResNet-34 (U-Net decoder)** | Baseline segmentation model | Lacks deep supervision, weaker edge precision |
| **YOLOv8** | Real-time object detector | Fast, but produces only bounding boxes |
| **Enhanced EfficientNet-B4 + U-Net** | Proposed hybrid model | Channel-attention skips + deep supervision, highest accuracy |

---

## ⚙️ Training Configuration
- **Optimizer:** AdamW  
- **Learning Rate:** Cosine-decay with warm-up  
- **Loss Function:** Binary Cross-Entropy + Dice Loss  
- **Batch Size:** 8  
- **Precision:** Mixed (AMP)  
- **Early Stopping:** Enabled  

---

## 📊 Evaluation Metrics
| Metric | Description |
|---------|--------------|
| **Accuracy** | Overall correct pixel predictions |
| **Dice Coefficient** | Overlap between predicted and true masks |
| **IoU (Intersection over Union)** | Common area ratio between prediction and ground truth |

**Results (Validation):**
| Model | Accuracy | Dice | IoU |
|--------|-----------|------|-----|
| ResNet | 91% | 0.67 | 0.54 |
| YOLOv8 | 87% | 0.47 | 0.41 |
| **Proposed Model** | **94%** | **0.74** | **0.62** |

---

## 🖼️ Qualitative Results
- **ResNet:** coarse, incomplete detection  
- **YOLOv8:** rough bounding-box localization only  
- **Proposed Model:** precise pothole boundaries matching ground truth  

---

## 💡 Applications
- Smart-city infrastructure monitoring  
- Vehicle-mounted road inspection systems  
- Municipal repair planning & prioritization  
- Extension to detect cracks, depressions, and patches  

---

## ⚠️ Limitations
- Dataset limited to few geographic regions  
- Reflection and lighting artifacts may cause false positives  
- EfficientNet-B4 backbone is heavy for embedded systems (requires pruning/quantization)

---

## 🚀 Future Work
- Expand dataset across diverse conditions and locations  
- Integrate stereo/IMU sensors for depth & volume estimation  
- Develop lightweight, mobile-optimized versions  
- Explore transformer-based global-context models  

---

## 🧩 Tech Stack
- **Languages:** Python 3.10  
- **Frameworks:** PyTorch / Torchvision  
- **Tools:** NumPy, OpenCV, Matplotlib, Albumentations, YOLOv8 (Ultralytics)  
- **Hardware:** NVIDIA GPU (CUDA 11+) recommended  

---

## 🧑‍💻 Contributors
| Name | Role |
|------|------|
| **Manikandan G.** | Model Design & Implementation |
| **Dhinesh Kumar S.** | Dataset Curation & Evaluation |
| **Dr. G. Revathy** | Project Guide |

---

## 📚 References
Key related works from *Sensors (2023–2025)*, *MDPI Electronics*, *arXiv preprints*, and *Automation in Construction* covering YOLO, CNN, and transformer-based damage detection.

---

## 🏁 Conclusion
The **Enhanced EfficientNet-B4 + U-Net** outperformed YOLOv8 and ResNet in Dice (0.74) and IoU (0.62), achieving state-of-the-art segmentation accuracy.  
This framework provides a **balanced, efficient**, and **scalable** solution for automated road-damage detection in smart-city infrastructure.

---

> 🧾 *Developed as part of Final Year Project, Department of AI & ML, VISTAS (2025).*
