"""
=============================================================
Zenodo Hand Hygiene Dataset - Full XGBoost Training Pipeline
=============================================================
Author: Ahmed (Cardiff Met Dissertation 2026)

Pipeline:
  1. For each DataSet (1-11), for each video (.mp4):
       - Read annotation CSV (consensus across Annotator1-4)
       - Extract MediaPipe hand landmarks frame by frame
       - Match each frame to its label via frame_time
       - Append 63 features + label to master dataset
  2. Train XGBoost with SMOTE (same as your best model)
  3. Evaluate and compare vs your previous results

Run in env_xgboost:
  cd C:\\dissertation
  conda activate env_xgboost   (or: env_xgboost\\Scripts\\activate)
  python train_zenodo_xgboost.py

Output files saved to C:\\dissertation\\
  - zenodo_landmarks.csv       (extracted features, reusable)
  - xgb_model_zenodo.pkl       (trained model)
  - scaler_zenodo.pkl          (scaler)
  - zenodo_results.txt         (accuracy report)
=============================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from collections import Counter

import mediapipe as mp
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# CONFIGURATION — adjust paths if needed
# ─────────────────────────────────────────────
DATASET_ROOT   = r"C:\Zenodo0 H H detection\4537209"
OUTPUT_DIR     = r"C:\dissertation"
LANDMARKS_CSV  = os.path.join(OUTPUT_DIR, "zenodo_landmarks.csv")
MODEL_PATH     = os.path.join(OUTPUT_DIR, "xgb_model_zenodo.pkl")
SCALER_PATH    = os.path.join(OUTPUT_DIR, "scaler_zenodo.pkl")
RESULTS_PATH   = os.path.join(OUTPUT_DIR, "zenodo_results.txt")

# Annotator folders to use for consensus
ANNOTATORS     = ["Annotator1", "Annotator2", "Annotator3", "Annotator4"]

# Only process datasets with this camera suffix
CAMERA_SUFFIX  = "camera102"

# MediaPipe: process every Nth frame to speed up extraction
# 1 = every frame (slowest, most data), 3 = every 3rd frame (recommended)
FRAME_SKIP     = 3

# Skip landmark extraction if CSV already exists (saves hours on re-run)
SKIP_EXTRACTION_IF_EXISTS = True

# ─────────────────────────────────────────────
# WHO step label names for reporting
# ─────────────────────────────────────────────
STEP_NAMES = {
    0: "No washing / Other",
    1: "Palm to palm",
    2: "Palm over dorsum",
    3: "Fingers interlaced",
    4: "Backs of fingers",
    5: "Rotational thumb",
    6: "Fingertips to palm",
    7: "Wrist / faucet off",
}

# ─────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands


def extract_features(hand_landmarks):
    """Convert 21 MediaPipe landmarks to 63 normalised features (x,y,z * 21)."""
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features  # length = 63


def load_annotation_consensus(dataset_path, video_stem):
    """
    Load annotation CSVs from all available annotators for a given video.
    Returns a dict: {frame_time_ms -> movement_code} using majority vote.
    """
    all_annotations = []

    for ann in ANNOTATORS:
        csv_path = os.path.join(dataset_path, "Annotations", ann, f"{video_stem}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if "frame_time" in df.columns and "movement_code" in df.columns:
                    all_annotations.append(df[["frame_time", "movement_code"]])
            except Exception:
                pass

    if not all_annotations:
        return {}

    # Build consensus: for each frame_time, take majority vote across annotators
    # Use first annotator's frame_times as reference
    ref_times = all_annotations[0]["frame_time"].values
    consensus = {}

    for ft in ref_times:
        votes = []
        for ann_df in all_annotations:
            row = ann_df[ann_df["frame_time"] == ft]
            if not row.empty:
                votes.append(int(row["movement_code"].values[0]))
        if votes:
            majority = Counter(votes).most_common(1)[0][0]
            consensus[float(ft)] = majority

    return consensus


def get_label_for_frame(frame_pos_ms, annotation_dict):
    """Find the closest annotated frame_time to the current video position."""
    if not annotation_dict:
        return None
    times = np.array(list(annotation_dict.keys()))
    closest_idx = np.argmin(np.abs(times - frame_pos_ms))
    closest_time = times[closest_idx]
    # Only accept if within 2 frame intervals (~67ms at 30fps)
    if abs(closest_time - frame_pos_ms) < 100:
        return annotation_dict[closest_time]
    return None


# ─────────────────────────────────────────────
# STEP 1: LANDMARK EXTRACTION
# ─────────────────────────────────────────────
def extract_all_landmarks():
    print("\n" + "="*60)
    print("STEP 1: Extracting landmarks from all videos")
    print("="*60)

    if SKIP_EXTRACTION_IF_EXISTS and os.path.exists(LANDMARKS_CSV):
        print(f"  Found existing {LANDMARKS_CSV} — skipping extraction.")
        print("  Delete the file to re-extract.")
        return

    col_names = [f"x{i}" if i % 3 == 0 else f"y{i}" if i % 3 == 1 else f"z{i}"
                 for i in range(63)]
    col_names = [f"lm{i//3}_{['x','y','z'][i%3]}" for i in range(63)]
    col_names += ["label", "dataset", "video"]

    rows = []
    total_frames = 0
    total_videos = 0
    skipped_videos = 0

    dataset_dirs = sorted([
        d for d in Path(DATASET_ROOT).iterdir()
        if d.is_dir() and d.name.startswith("DataSet")
    ])

    print(f"  Found {len(dataset_dirs)} DataSet folders.\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for ds_dir in dataset_dirs:
            video_dir = ds_dir / "Videos"
            if not video_dir.exists():
                continue

            videos = sorted(video_dir.glob(f"*{CAMERA_SUFFIX}*.mp4"))
            print(f"  [{ds_dir.name}] {len(videos)} videos found...")

            for vid_path in videos:
                video_stem = vid_path.stem  # e.g. 2020-06-26_18-28-10_camera102

                # Load annotation consensus
                ann = load_annotation_consensus(str(ds_dir), video_stem)
                if not ann:
                    skipped_videos += 1
                    continue

                cap = cv2.VideoCapture(str(vid_path))
                if not cap.isOpened():
                    skipped_videos += 1
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_idx = 0
                video_frames = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % FRAME_SKIP != 0:
                        continue

                    # Current position in milliseconds
                    frame_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    label = get_label_for_frame(frame_ms, ann)

                    if label is None:
                        continue

                    # MediaPipe detection
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    if result.multi_hand_landmarks:
                        # Use first detected hand
                        feats = extract_features(result.multi_hand_landmarks[0])
                        rows.append(feats + [label, ds_dir.name, video_stem])
                        video_frames += 1

                cap.release()
                total_frames += video_frames
                total_videos += 1

                if total_videos % 10 == 0:
                    print(f"    Processed {total_videos} videos, "
                          f"{total_frames:,} frames so far...")

            print(f"  [{ds_dir.name}] done. Running total: {total_frames:,} frames")

    print(f"\n  Extraction complete!")
    print(f"  Total videos processed : {total_videos}")
    print(f"  Skipped (no annotation): {skipped_videos}")
    print(f"  Total frames extracted : {total_frames:,}")

    if not rows:
        print("  ERROR: No landmarks extracted. Check paths and MediaPipe installation.")
        return

    # Save to CSV
    df = pd.DataFrame(rows, columns=col_names)
    df.to_csv(LANDMARKS_CSV, index=False)
    print(f"\n  Saved: {LANDMARKS_CSV}")
    print(f"  Shape: {df.shape}")

    # Label distribution
    print("\n  Label distribution:")
    for code, count in sorted(df["label"].value_counts().items()):
        name = STEP_NAMES.get(int(code), "Unknown")
        print(f"    Step {code} ({name}): {count:,} frames")


# ─────────────────────────────────────────────
# STEP 2: TRAIN XGBOOST
# ─────────────────────────────────────────────
def train_model():
    print("\n" + "="*60)
    print("STEP 2: Training XGBoost on Zenodo dataset")
    print("="*60)

    if not os.path.exists(LANDMARKS_CSV):
        print("  ERROR: Landmarks CSV not found. Run extraction first.")
        return

    print(f"  Loading {LANDMARKS_CSV}...")
    df = pd.read_csv(LANDMARKS_CSV)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Feature columns (63 landmark features)
    feat_cols = [c for c in df.columns if c.startswith("lm")]
    X = df[feat_cols].values
    y = df["label"].values.astype(int)

    print(f"\n  Feature shape : {X.shape}")
    print(f"  Label distribution:")
    for code, count in sorted(Counter(y).items()):
        name = STEP_NAMES.get(code, "Unknown")
        pct = count / len(y) * 100
        print(f"    Step {code} ({name}): {count:,} ({pct:.1f}%)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Scale
    print("\n  Scaling features...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE for class balance
    print("  Applying SMOTE...")
    try:
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
        print(f"  After SMOTE: {len(X_train_sm):,} samples")
    except Exception as e:
        print(f"  SMOTE skipped ({e}) — using original distribution")
        X_train_sm, y_train_sm = X_train_sc, y_train

    # Train XGBoost (same hyperparameters as your best model)
    print("\n  Training XGBoost...")
    print("  (This may take 10-30 minutes on large dataset — please wait...)")
    t0 = time.time()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,        # use all CPU cores
        tree_method="hist"  # faster on CPU
    )
    model.fit(X_train_sm, y_train_sm)
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed/60:.1f} minutes")

    # Evaluate
    print("\n  Evaluating...")
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  {'='*40}")
    print(f"  ZENODO DATASET ACCURACY: {acc*100:.2f}%")
    print(f"  {'='*40}")

    # Comparison table
    print("\n  COMPARISON WITH YOUR PREVIOUS RESULTS:")
    print(f"  {'Model':<35} {'Accuracy':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Logistic Regression (3-view)':<35} {'~81.00%':>10}")
    print(f"  {'SVM (3-view)':<35} {'~87.00%':>10}")
    print(f"  {'CNN+Transformer (3-view)':<35} {'~93.00%':>10}")
    print(f"  {'CNN+LSTM (3-view)':<35} {'~94.00%':>10}")
    print(f"  {'XGBoost + SMOTE (3-view)':<35} {'~97.00%':>10}")
    print(f"  {'XGBoost Personalised (your data)':<35} {'96.75%':>10}")
    print(f"  {'XGBoost (Zenodo full dataset)':<35} {f'{acc*100:.2f}%':>10}  ← NEW")

    report = classification_report(y_test, y_pred,
                                   target_names=[f"Step {i}: {STEP_NAMES[i]}"
                                                 for i in sorted(STEP_NAMES.keys())])
    print(f"\n  Classification Report:\n{report}")

    # Save model and scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved model  : {MODEL_PATH}")
    print(f"  Saved scaler : {SCALER_PATH}")

    # Save results to text file
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Zenodo Full Dataset XGBoost Results\n")
        f.write(f"====================================\n")
        f.write(f"Total frames : {len(df):,}\n")
        f.write(f"Train samples: {len(X_train_sm):,} (after SMOTE)\n")
        f.write(f"Test samples : {len(X_test):,}\n")
        f.write(f"Accuracy     : {acc*100:.2f}%\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"\nComparison:\n")
        f.write(f"  Logistic Regression (3-view): ~81.00%\n")
        f.write(f"  SVM (3-view):                ~87.00%\n")
        f.write(f"  CNN+Transformer (3-view):    ~93.00%\n")
        f.write(f"  CNN+LSTM (3-view):           ~94.00%\n")
        f.write(f"  XGBoost SMOTE (3-view):      ~97.00%\n")
        f.write(f"  XGBoost Personalised:         96.75%\n")
        f.write(f"  XGBoost Zenodo (full):        {acc*100:.2f}%\n")
    print(f"  Saved results: {RESULTS_PATH}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Zenodo Hand Hygiene - XGBoost Training Pipeline")
    print("  Cardiff Met Dissertation 2026")
    print("="*60)

    # Check dataset root exists
    if not os.path.exists(DATASET_ROOT):
        print(f"\nERROR: Dataset root not found:\n  {DATASET_ROOT}")
        print("Please check the DATASET_ROOT path at the top of this script.")
        exit(1)

    start = time.time()

    extract_all_landmarks()
    train_model()

    total_time = (time.time() - start) / 60
    print(f"\n{'='*60}")
    print(f"  All done! Total time: {total_time:.1f} minutes")
    print(f"  Results saved to: {RESULTS_PATH}")
    print(f"{'='*60}\n")
