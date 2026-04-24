# Automated Computer Vision System for Real-Time WHO 7-Step Hand Hygiene Compliance Tracking

**MSc Dissertation — ENG7003 | Cardiff Metropolitan University | 2026**  
**Author:** Salah Mohamed Mosalam | Student ID: st20312571  
**Supervisor:** Dr. Imran Baig

---

## Project Overview

This dissertation presents an automated computer vision system for 
real-time WHO 7-step hand hygiene compliance tracking using CPU-only 
landmark-based machine learning — no GPU required.

The system employs Google MediaPipe for hand landmark extraction, 
yielding 126 three-dimensional features per frame (21 landmarks × 3 
coordinates × 2 hands), classified using machine learning.

### Key Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 80.72% |
| SVM (baseline) | ~87% |
| CNN+Transformer | 92.80% |
| CNN+LSTM | 93.94% |
| XGBoost + SMOTE | 96.58% |
| XGBoost + Personalised | **96.75%** |
| XGBoost + Orbbec RGB-D | 95.8% |

- ✅ 100% real-time step completion on both camera configurations
- ✅ No GPU required — runs entirely on CPU
- ✅ Average scan time: 2.07s/step (laptop) | 2.59s/step (Orbbec)
- ✅ Generalisation experiment: XGBoost 59.53% → YOLOv8+BiLSTM ~72% on Zenodo clinical data

---

## System Architecture
Camera Input (RGB 30FPS or Orbbec RGB-D 10FPS)
↓
MediaPipe Hands (21 landmarks × 3D × 2 hands = 126 features)
↓
StandardScaler Normalisation
↓
XGBoost Classifier (500 trees, 7 WHO step classes)
↓
5-frame Vote Buffer + 1.5s Voting Window
↓
Real-time Step Display + Compliance Report
---

## Repository Structure

| File | Description |
|---|---|
| `webcam_xgboost.py` | Real-time detection — laptop RGB camera |
| `webcam_orbbec.py` | Real-time detection — Orbbec Gemini 336L RGB-D |
| `webcam_cnn_lstm.py` | Real-time detection — CNN+LSTM model |
| `webcam_cnn_transformer.py` | Real-time detection — CNN+Transformer model |
| `landmark_helper_with_frame.py` | Landmark extraction helper for CNN+LSTM |
| `collect_my_data.py` | Personalisation data collection — laptop camera |
| `collect_my_data_orbbec.py` | Personalisation data collection — Orbbec camera |
| `train_zenodo_xgboost.py` | XGBoost training on Zenodo clinical dataset |
| `generate_real_landmark_plot.py` | Landmark distribution visualisation |
| `test_orbbec.py` | Orbbec camera testing script |
| `xgb_model_personal.pkl` | XGBoost personalised model — 96.75% accuracy |
| `xgb_model_smote.pkl` | XGBoost SMOTE model — 96.58% accuracy |
| `xgb_model_orbbec.pkl` | XGBoost Orbbec model — 95.8% accuracy |
| `xgb_model_zenodo.pkl` | XGBoost Zenodo generalisation model |
| `scaler.pkl` | Pre-fitted StandardScaler for real-time deployment |
| `cnn_lstm_model.h5` | CNN+LSTM trained model — 93.94% accuracy |
| `cnn_transformer_model.h5` | CNN+Transformer trained model — 92.80% accuracy |
| `yolo_bilstm/` | YOLOv8+BiLSTM temporal pipeline — Zenodo dataset |

---

## Installation

### Requirements
- Python 3.11
- Two virtual environments required due to protobuf conflicts

### Environment 1 — env_xgboost (for XGBoost and Orbbec)
```bash
pip install mediapipe==0.10.9
pip install protobuf==3.20.3
pip install xgboost
pip install scikit-learn==1.6.1
pip install opencv-python
pip install numpy pandas
```

### Environment 2 — env_tensorflow (for CNN models)
```bash
pip install tensorflow-cpu
pip install scikit-learn==1.6.1
pip install opencv-python
pip install numpy pandas
```

### Orbbec Camera SDK
```bash
pip install pyorbbecsdk2
```

---

## How to Run

### Real-time detection — Laptop RGB camera
```bash
# Activate env_xgboost first
python webcam_xgboost.py
```

### Real-time detection — Orbbec RGB-D camera
```bash
# Activate env_xgboost first
python webcam_orbbec.py
```

### Real-time detection — CNN+LSTM
```bash
# Activate env_tensorflow first
python webcam_cnn_lstm.py
```

---

## Datasets

| Dataset | Description | Access |
|---|---|---|
| SunnySideUp11 Hand-Hygiene-ICU | 19,959 landmark frames, 8 subjects, 3 camera views | [GitHub](https://github.com/SunnySideUp11/Hand-Hygiene-ICU) |
| Zenodo Hand-Washing Videos | 3,185 clinical episodes, Pauls Stradins Hospital | [DOI: 10.5281/zenodo.4537209](https://doi.org/10.5281/zenodo.4537209) |

---

## Hardware

| Component | Specification |
|---|---|
| Primary camera | Laptop built-in webcam — RGB, 30 FPS |
| Depth camera | Orbbec Gemini 336L RGB-D — configured at 10 FPS |
| Processing | CPU only — no GPU required |
| OS | Windows 11 |
| Python | 3.11 |

---

## Citation

If you use this work please cite:Mosalam, S. (2026) Automated Computer Vision System for Real-Time
WHO 7-Step Hand Hygiene Compliance Tracking. MSc Dissertation,
Cardiff Metropolitan University.

---

## License

This project is released for academic use under the Cardiff Metropolitan 
University ENG7003 dissertation module requirements.

---

## Contact

**Salah Mohamed Mosalam**  
Cardiff Metropolitan University  
st20312571@cardiffmet.ac.uk
