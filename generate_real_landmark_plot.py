"""
Generate REAL landmark distribution plot from your actual CSV data
==================================================================
Run this in env_xgboost:
    cd C:\\dissertation
    env_xgboost\\Scripts\\activate
    python generate_real_landmark_plot.py

Output: landmark_distribution_REAL.png
Save this and send it to replace the fake one in your dissertation!
==================================================================
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ────────────────────────────────────────────────
# Point this to your ICU landmark CSV folder
DATA_DIR = r"C:\dissertation"

# Try common locations for the landmark CSV files
POSSIBLE_PATHS = [
    r"C:\dissertation\Hand-Hygiene-ICU",
    r"C:\dissertation\data",
    r"C:\dissertation\landmark_data",
    r"C:\dissertation\dataset",
]

# Step names
STEP_NAMES = {
    0: 'Step 1\nPalm to palm',
    1: 'Step 2\nRight over left',
    2: 'Step 3\nFingers interlaced',
    3: 'Step 4\nBacks of fingers',
    4: 'Step 5\nRotational thumb',
    5: 'Step 6\nFingertips',
    6: 'Step 7\nWrists',
}

# ── LOAD DATA ─────────────────────────────────────────────
def parse_landmarks(val):
    """Parse a JSON-style landmark string into a numpy array."""
    try:
        if isinstance(val, str) and val.strip():
            arr = ast.literal_eval(val)
            return np.array(arr, dtype=np.float32).flatten()
    except Exception:
        pass
    return np.zeros(63, dtype=np.float32)

def load_csv_files(data_dir):
    """Load all CSV files from directory and extract features + labels."""
    all_X = []
    all_y = []

    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None, None

    print(f"Found {len(csv_files)} CSV files")

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            print(f"  Loading: {os.path.basename(fpath)} — shape {df.shape}")

            # Detect label column
            label_col = None
            for col in ['label', 'Label', 'step', 'Step', 'class', 'Class', 'y']:
                if col in df.columns:
                    label_col = col
                    break

            if label_col is None:
                # Try last column
                label_col = df.columns[-1]
                print(f"    Using last column as label: {label_col}")

            # Detect landmark columns
            left_col = right_col = None
            for col in df.columns:
                cl = col.lower()
                if 'left' in cl and left_col is None:
                    left_col = col
                if 'right' in cl and right_col is None:
                    right_col = col

            if left_col is None or right_col is None:
                print(f"    Could not find Left/Right columns — skipping")
                continue

            for _, row in df.iterrows():
                left  = parse_landmarks(row[left_col])
                right = parse_landmarks(row[right_col])
                feat  = np.concatenate([left, right])
                if len(feat) == 126:
                    label = int(row[label_col])
                    # Normalise label to 0-6
                    if label > 0:
                        label -= 1
                    all_X.append(feat)
                    all_y.append(label)

        except Exception as e:
            print(f"    Error loading {fpath}: {e}")
            continue

    if not all_X:
        return None, None

    return np.array(all_X), np.array(all_y)


# ── PLOT ──────────────────────────────────────────────────
def plot_landmark_distribution(X, y, output_path='landmark_distribution_REAL.png'):
    """Plot left-hand vs right-hand landmark distributions per step."""

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.flatten()

    fig.suptitle(
        'Distribution of Left-Hand and Right-Hand Landmark Points\n'
        'Across the Seven WHO Hand Hygiene Steps',
        fontsize=14, fontweight='bold', y=0.98
    )

    unique_steps = sorted(np.unique(y))
    n_steps = len(unique_steps)
    print(f"\nFound {n_steps} unique steps: {unique_steps}")

    for i, step in enumerate(unique_steps):
        if i >= 9:
            break
        ax = axes[i]
        mask = y == step
        step_data = X[mask]

        # Left hand: first 63 features → x coords are indices 0,3,6,...
        # x = landmark index 0 (normalised x of landmark 0)
        # We plot landmark x vs y for all 21 landmarks
        left_feats  = step_data[:, :63]   # left hand 63 features
        right_feats = step_data[:, 63:]   # right hand 63 features

        # Extract x,y coords (every 3rd value starting at 0 and 1)
        left_x  = left_feats[:,  ::3].flatten()  # x coords
        left_y  = left_feats[:, 1::3].flatten()  # y coords
        right_x = right_feats[:, ::3].flatten()
        right_y = right_feats[:,1::3].flatten()

        # Filter out zeros (missing hand)
        lmask = (left_x != 0) | (left_y != 0)
        rmask = (right_x != 0) | (right_y != 0)

        # Sample for speed
        n_plot = min(500, lmask.sum(), rmask.sum())
        li = np.random.choice(np.where(lmask)[0], n_plot, replace=False) if lmask.sum() >= n_plot else np.where(lmask)[0]
        ri = np.random.choice(np.where(rmask)[0], n_plot, replace=False) if rmask.sum() >= n_plot else np.where(rmask)[0]

        ax.scatter(left_x[li],  left_y[li],  c='#D85A30', alpha=0.3, s=6, label='Left hand')
        ax.scatter(right_x[ri], right_y[ri], c='#185FA5', alpha=0.3, s=6, label='Right hand')

        step_name = STEP_NAMES.get(step, f'Step {step+1}')
        ax.set_title(step_name, fontsize=10, fontweight='bold', pad=3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Normalised x', fontsize=8)
        ax.set_ylabel('Normalised y', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        n_samples = mask.sum()
        ax.text(0.98, 0.98, f'n={n_samples:,}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color='gray')

    # Hide unused subplots
    for j in range(n_steps, 9):
        axes[j].set_visible(False)

    # Legend
    red_patch  = mpatches.Patch(color='#D85A30', label='Left hand landmarks')
    blue_patch = mpatches.Patch(color='#185FA5', label='Right hand landmarks')
    fig.legend(handles=[red_patch, blue_patch],
               loc='lower right', bbox_to_anchor=(0.98, 0.02),
               fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {output_path}")
    return output_path


# ── MAIN ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Real Landmark Distribution Generator")
    print("  Cardiff Met Dissertation 2026 — Ahmed")
    print("=" * 55)

    # Try to find data
    X, y = None, None
    search_paths = POSSIBLE_PATHS + [DATA_DIR]

    for path in search_paths:
        if os.path.exists(path):
            print(f"\nSearching: {path}")
            X, y = load_csv_files(path)
            if X is not None:
                print(f"Loaded {len(X)} frames successfully!")
                break

    if X is None:
        print("\nCould not find CSV files automatically.")
        print("Please set DATA_DIR at the top of this script to your")
        print("Hand-Hygiene-ICU folder path and run again.")
    else:
        print(f"\nData shape: X={X.shape}, y={y.shape}")
        print(f"Labels: {np.unique(y)}")
        print("\nGenerating plot...")
        out = plot_landmark_distribution(X, y)
        print(f"\nDone! Upload {out} to replace the fake figure in your dissertation.")
