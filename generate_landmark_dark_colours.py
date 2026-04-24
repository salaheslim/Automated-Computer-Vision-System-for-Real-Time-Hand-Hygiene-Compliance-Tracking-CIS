"""
Regenerate landmark distribution with dark vivid colours
=========================================================
Run in env_xgboost:
    cd C:\dissertation
    env_xgboost\Scripts\activate
    python generate_landmark_dark_colours.py
=========================================================
"""

import os, ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ─────────────────────────────────────────────────
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

# ── DARK VIVID COLOURS ─────────────────────────────────────
LEFT_COLOR  = '#B71C1C'   # dark red
RIGHT_COLOR = '#0D2D6B'   # dark navy blue

def parse_lm(val):
    try:
        if isinstance(val, str) and val.strip().startswith('['):
            arr = ast.literal_eval(val.strip())
            return np.array(arr, dtype=np.float32).flatten()
    except:
        pass
    return np.zeros(63, dtype=np.float32)

def load_icu_data(view_dirs):
    all_lx = {i: [] for i in range(1, 8)}
    all_ly = {i: [] for i in range(1, 8)}
    all_rx = {i: [] for i in range(1, 8)}
    all_ry = {i: [] for i in range(1, 8)}
    total  = 0

    for view_dir in view_dirs:
        if not os.path.exists(view_dir):
            continue
        print(f"Scanning: {view_dir}")
        for root, dirs, files in os.walk(view_dir):
            for fname in files:
                if not fname.endswith('.csv'):
                    continue
                fpath = os.path.join(root, fname)
                parts = root.replace('\\', '/').split('/')
                step  = None
                try:
                    step = int(parts[-1])
                    if not 1 <= step <= 7:
                        step = None
                except:
                    pass
                if step is None:
                    try:
                        step = int(fname.replace('.csv', ''))
                        if not 1 <= step <= 7:
                            step = None
                    except:
                        pass
                if step is None:
                    continue
                try:
                    import pandas as pd
                    df = pd.read_csv(fpath, header=0)
                    lc = rc = None
                    for col in df.columns:
                        cl = str(col).lower().strip()
                        if cl == 'left':
                            lc = col
                        elif cl == 'right':
                            rc = col
                    if lc is None or rc is None:
                        continue
                    for _, row in df.iterrows():
                        lm_l = parse_lm(str(row[lc]))
                        lm_r = parse_lm(str(row[rc]))
                        if len(lm_l) == 63 and len(lm_r) == 63:
                            lx = lm_l[0::3]; ly = lm_l[1::3]
                            rx = lm_r[0::3]; ry = lm_r[1::3]
                            if np.any(lx != 0):
                                all_lx[step].extend(lx.tolist())
                                all_ly[step].extend(ly.tolist())
                            if np.any(rx != 0):
                                all_rx[step].extend(rx.tolist())
                                all_ry[step].extend(ry.tolist())
                            total += 1
                except:
                    continue

    print(f"Total frames: {total}")
    return all_lx, all_ly, all_rx, all_ry

def plot_distribution(all_lx, all_ly, all_rx, all_ry,
                      out='landmark_distribution_DARK.png'):

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()
    fig.patch.set_facecolor('white')

    fig.suptitle(
        'Distribution of Left-Hand and Right-Hand Landmark Points\n'
        'Across the Seven WHO Hand Hygiene Steps',
        fontsize=15, fontweight='bold', y=0.98
    )

    for i, step in enumerate(range(1, 8)):
        ax = axes[i]
        ax.set_facecolor('#FAFAFA')

        lx = np.array(all_lx[step])
        ly = np.array(all_ly[step])
        rx = np.array(all_rx[step])
        ry = np.array(all_ry[step])

        n = min(4000, len(lx), len(rx))
        if n == 0:
            ax.set_title(f'Step {step} — no data', fontsize=9)
            continue

        li = np.random.choice(len(lx), n, replace=False)
        ri = np.random.choice(len(rx), n, replace=False)

        # Plot right hand first (behind), then left hand on top
        ax.scatter(rx[ri], ry[ri],
                   c=RIGHT_COLOR, alpha=0.55, s=7,
                   label='Right hand', edgecolors='none', zorder=2)
        ax.scatter(lx[li], ly[li],
                   c=LEFT_COLOR,  alpha=0.60, s=7,
                   label='Left hand',  edgecolors='none', zorder=3)

        name = STEP_NAMES.get(step, f'Step {step}')
        ax.set_title(name, fontsize=11, fontweight='bold', pad=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Normalised x', fontsize=9)
        ax.set_ylabel('Normalised y', fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['left'].set_color('#cccccc')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

        n_frames_l = len(lx) // 21
        n_frames_r = len(rx) // 21
        ax.text(0.98, 0.98,
                f'L:{n_frames_l}  R:{n_frames_r}',
                transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                color='#888888')

    # Hide spare subplots
    axes[7].set_visible(False)
    axes[8].set_visible(False)

    # Legend
    red_p  = mpatches.Patch(color=LEFT_COLOR,  label='Left hand landmarks')
    blue_p = mpatches.Patch(color=RIGHT_COLOR, label='Right hand landmarks')
    fig.legend(handles=[red_p, blue_p],
               loc='lower right',
               bbox_to_anchor=(0.97, 0.03),
               fontsize=11, frameon=True,
               framealpha=0.9,
               edgecolor='#cccccc')

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig(out, dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")
    print("Upload this to Claude!")

if __name__ == '__main__':
    print("=" * 55)
    print("  Landmark Plot — Dark Vivid Colours")
    print("  Cardiff Met Dissertation 2026")
    print("=" * 55)
    np.random.seed(42)
    lx, ly, rx, ry = load_icu_data(ICU_VIEWS)
    total = sum(len(lx[s]) for s in range(1,8)) // 21
    if total == 0:
        print("No data found. Check view1/view2/view3 paths.")
    else:
        print(f"\nPlotting {total} frames...")
        plot_distribution(lx, ly, rx, ry)
