
# 🫁 PulmoScan: An AI-Powered Diagnostic Pipeline for Lung Cancer Detection and Staging

## 📄 Abstract
**PulmoScan** is an AI-integrated diagnostic system that enhances the early detection, classification, and staging of lung cancer using deep learning and medical imaging.  
It consists of:
- 🧪 An **early prediction model** based on clinical data.
- 🧠 A **multi-stage imaging pipeline** for:
  - Nodule detection
  - Malignancy classification
  - Cancer staging  

PulmoScan uses public datasets such as `LIDC-IDRI`, `LUNA16`, and `NSCLC-Radiomics`, and is deployed through a user-friendly **Django-based web interface** to assist radiologists with real-time diagnosis.

---

## 🔍 1. Introduction
Lung cancer is the leading cause of cancer-related deaths globally. Traditional diagnostics face challenges:
- High workloads for radiologists
- Variability in interpretation
- Delays in diagnosis

**PulmoScan** addresses these issues through AI-based automation, enabling faster and more accurate analysis of CT scans and clinical data.

---

## 🧪 2. Methodology

### 📊 2.1 Datasets
| Task | Dataset |
|------|---------|
| Early Prediction | Kaggle Lung Cancer (clinical & lifestyle) |
| Nodule Detection | LUNA16 (from LIDC-IDRI) |
| Nodule Classification | LIDC-IDRI |
| Malignant Subtype Classification | Kaggle Chest CT Dataset |
| Staging | NSCLC-Radiomics |

### 🧼 2.2 Preprocessing
- Normalization & resampling (1×1×1 mm)
- Windowing (-600 HU center)
- Data augmentation
- Clinical feature encoding and scaling

### 🧠 2.3 Model Architecture
| Task | Model |
|------|-------|
| Early Detection | Decision Tree (for interpretability) |
| Nodule Detection | EfficientNetB0 |
| Nodule Classification | 3D CNN |
| Malignant Subtype Classification | YOLOv8 |
| Staging | VGG-Style CNN |

---

## 📊 3. Results

| Task | Model | Accuracy | F1-Score | AUC |
|------|-------|----------|----------|-----|
| Early Prediction | Decision Tree | ~94% | 0.94 | - |
| Nodule Detection | EfficientNetB0 | 92.96% | 94.07% | 0.98 |
| Nodule Classification | 3D CNN | 76.84% | 0.54 (macro) | - |
| Malignant Classification | YOLOv8 | 85.7% | 0.86 (macro) | - |
| Staging | VGG-Style CNN | 93% | >90% | - |

🟢 The pipeline showed **high accuracy**, especially in nodule detection and subtype classification.

---

## 🚀 4. Deployment

The system is **fully deployed using Django** and includes:
- 👤 **Patient Module**: Early cancer risk prediction from clinical data
- 🧠 **Diagnostic Pipeline**: CT image analysis through 4 stages:
  - Cancer detection
  - Benign/malignant classification
  - Subtype recognition
  - Cancer staging
- 📊 **Doctor Dashboard**: Case history, predictions, monitoring

---

## 💬 5. Discussion

PulmoScan combines CNN models with structured data to provide accurate, explainable lung cancer diagnosis.  
📉 **Limitations**:
- Class imbalance
- Limited subtype data

But the system proves strong generalization and potential for clinical use.

---

## ✅ 6. Conclusion

PulmoScan is:
- ✔️ Practical
- ✔️ Interpretable
- ✔️ Scalable  

📌 Future directions:
- Real-time hospital integration
- Larger and more diverse datasets
- Multilingual interface
