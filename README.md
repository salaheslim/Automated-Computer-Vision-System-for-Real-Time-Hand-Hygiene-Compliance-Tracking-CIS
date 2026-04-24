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
