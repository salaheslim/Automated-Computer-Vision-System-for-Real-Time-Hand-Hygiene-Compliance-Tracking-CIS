"""
Real Landmark Distribution Plot — FIXED for ICU CSV format
===========================================================
Run in env_xgboost:
    cd C:\dissertation
    env_xgboost\Scripts\activate
    python generate_real_landmark_plot_v2.py

Output: landmark_distribution_REAL.png
Upload this to Claude to replace the fake figure!
===========================================================
"""

import os, ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ────────────────────────────────────────────────
# ICU dataset folders (view1, view2, view3 inside C:\dissertation)
ICU_VIEWS = [
    r"C:\dissertation\view1",
    r"C:\dissertation\view2",
    r"C:\dissertation\view3",
]

STEP_NAMES = {
    1: 'Step 1\nPalm to palm',
    2: 'Step 2\nRight over left',
    3: 'Step 3\nFingers interlaced',
    4: 'Step 4\nBacks of fingers',
    5: 'Step 5\nRotational thumb',
    6: 'Step 6\nFingertips',
    7: 'Step 7\nWrists',
}

# ── PARSE ONE LANDMARK STRING ──────────────────────────────
def parse_lm(val):
    """Parse '[x, y, z, x, y, z, ...]' string into numpy array."""
    try:
        if isinstance(val, str) and val.strip().startswith('['):
            arr = ast.literal_eval(val.strip())
            return np.array(arr, dtype=np.float32).flatten()
    except Exception:
        pass
    return np.zeros(63, dtype=np.float32)

# ── LOAD ICU DATA ─────────────────────────────────────────
def load_icu_data(view_dirs):
    """
    ICU CSV format:
      Folder structure: view1/1/1.csv, view1/1/2.csv ... (subject / step)
      OR:               view1/step1/subject1.csv
    CSV columns: [something, Left, Right]
      where Left and Right are JSON landmark arrays
    The STEP comes from the CSV filename or parent folder name.
    """
    all_left_x  = {i: [] for i in range(1, 8)}
    all_left_y  = {i: [] for i in range(1, 8)}
    all_right_x = {i: [] for i in range(1, 8)}
    all_right_y = {i: [] for i in range(1, 8)}
    total = 0

    for view_dir in view_dirs:
        if not os.path.exists(view_dir):
            print(f"  View not found: {view_dir}")
            continue

        print(f"\nScanning: {view_dir}")

        # Walk all subdirectories
        for root, dirs, files in os.walk(view_dir):
            for fname in files:
                if not fname.endswith('.csv'):
                    continue

                fpath = os.path.join(root, fname)

                # Determine step number from folder or filename
                # Structure: view1/{step_num}/{subject_num}.csv
                # OR:        view1/{subject_num}/{step_num}.csv
                parts = root.replace('\\', '/').split('/')
                step = None

                # Try: the immediate parent folder is the step number
                try:
                    step = int(parts[-1])
                    if not 1 <= step <= 7:
                        step = None
                except:
                    pass

                # Try: the filename (without .csv) is the step number
                if step is None:
                    try:
                        step = int(fname.replace('.csv', ''))
                        if not 1 <= step <= 7:
                            step = None
                    except:
                        pass

                if step is None:
                    continue  # can't determine step

                # Load CSV
                try:
                    import pandas as pd
                    df = pd.read_csv(fpath, header=0)

                    # Find Left and Right columns
                    left_col = right_col = None
                    for col in df.columns:
                        cl = str(col).lower().strip()
                        if cl == 'left':
                            left_col = col
                        elif cl == 'right':
                            right_col = col

                    if left_col is None or right_col is None:
                        continue

                    for _, row in df.iterrows():
                        lm_left  = parse_lm(str(row[left_col]))
                        lm_right = parse_lm(str(row[right_col]))

                        if len(lm_left) == 63 and len(lm_right) == 63:
                            # Extract x,y for all 21 landmarks
                            # Format: [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
                            lx = lm_left[0::3]   # x coords
                            ly = lm_left[1::3]    # y coords
                            rx = lm_right[0::3]
                            ry = lm_right[1::3]

                            # Only use non-zero (hand detected)
                            if np.any(lx != 0):
                                all_left_x[step].extend(lx.tolist())
                                all_left_y[step].extend(ly.tolist())
                            if np.any(rx != 0):
                                all_right_x[step].extend(rx.tolist())
                                all_right_y[step].extend(ry.tolist())
                            total += 1

                except Exception as e:
                    continue

    print(f"\nTotal frames loaded: {total}")
    for s in range(1, 8):
        print(f"  Step {s}: {len(all_left_x[s])//21} left frames, "
              f"{len(all_right_x[s])//21} right frames")

    return all_left_x, all_left_y, all_right_x, all_right_y

# ── PLOT ──────────────────────────────────────────────────
def plot_distribution(all_lx, all_ly, all_rx, all_ry,
                      out='landmark_distribution_REAL.png'):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.flatten()

    fig.suptitle(
        'Distribution of Left-Hand and Right-Hand Landmark Points\n'
        'Across the Seven WHO Hand Hygiene Steps',
        fontsize=14, fontweight='bold', y=0.98
    )

    for i, step in enumerate(range(1, 8)):
        ax = axes[i]
        lx = np.array(all_lx[step])
        ly = np.array(all_ly[step])
        rx = np.array(all_rx[step])
        ry = np.array(all_ry[step])

        n = min(3000, len(lx), len(rx))
        if n == 0:
            ax.set_title(f'Step {step} — no data', fontsize=9)
            continue

        li = np.random.choice(len(lx), n, replace=False)
        ri = np.random.choice(len(rx), n, replace=False)

        ax.scatter(lx[li], ly[li], c='#D85A30', alpha=0.25, s=5,
                   label='Left hand')
        ax.scatter(rx[ri], ry[ri], c='#185FA5', alpha=0.25, s=5,
                   label='Right hand')

        name = STEP_NAMES.get(step, f'Step {step}')
        ax.set_title(name, fontsize=10, fontweight='bold', pad=3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Normalised x', fontsize=8)
        ax.set_ylabel('Normalised y', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.98, 0.98,
                f'L:{len(lx)//21} R:{len(rx)//21}',
                transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color='gray')

    # Hide spare subplots
    axes[7].set_visible(False)
    axes[8].set_visible(False)

    # Legend
    red_p  = mpatches.Patch(color='#D85A30', label='Left hand landmarks')
    blue_p = mpatches.Patch(color='#185FA5', label='Right hand landmarks')
    fig.legend(handles=[red_p, blue_p],
               loc='lower right', bbox_to_anchor=(0.98, 0.02),
               fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    print("Upload this PNG to Claude to replace the fake figure!")

# ── MAIN ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Real Landmark Distribution Generator v2")
    print("  Cardiff Met Dissertation 2026 — Ahmed")
    print("=" * 55)

    lx, ly, rx, ry = load_icu_data(ICU_VIEWS)

    total_frames = sum(len(lx[s]) for s in range(1,8)) // 21
    if total_frames == 0:
        print("\nNo data loaded. Check that view1/view2/view3 folders")
        print("exist inside C:\\dissertation and contain CSV files.")
    else:
        print(f"\nGenerating plot from {total_frames} real frames...")
        plot_distribution(lx, ly, rx, ry)
