# dLEM quick-start example
# ========================
# Loads a small H1-hESC Hi-C file (H1-hESC, chr10:19–26 Mb, 10 kb resolution),
# fits a dLEM model, and reproduces the reference locus contact map from the paper
# (chr10:20.5–22.5 Mb).
#
# Run from the repo root:
#   python docs/quick_start.py
#
# Output: docs/quick_start_patch.png
#
# Data source: 4DNFI9GMP2J8 (H1-hESC, 4D Nucleome).
# The example .cool stores this region under the chromosome name 'ref_region'
# with coordinates shifted to start at 0 (bin 0 = chr10:19,000,000).

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless rendering — no display required
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp

from dlem.api import fetch_band, train_dlem, flip_diag_row
from dlem.core import normalize_expected_observed, jax_forward_generate

# ── 1. Load data ─────────────────────────────────────────────────────────────

COOL       = os.path.join(os.path.dirname(__file__), 'data', 'example_chr10.cool')
REGION     = 'ref_region:0-7000000'
RESOLUTION = 10_000

# Offset for converting local bin positions to true chr10 coordinates (Mb)
GENOMIC_OFFSET_MB = 19.0

band = fetch_band(COOL, RESOLUTION, REGION, width=700)  # (700, 700), float32
band_train = band[1:170, :]                              # rows 1–169 (exclude row 0)

# ── 2. Train dLEM (paper settings) ───────────────────────────────────────────

result = train_dlem(
    band_train,
    steps=10,
    start_row=5,
    slowdown=0.025,
    learning_rate=1e-2,
    train_steps=300,
    loss_type='multinomial',
    weight_power=0,
    auto_stop_metric='mse',
    verbose=True,
)
p_left  = np.array(result['p_left_mse'])
p_right = np.array(result['p_right_mse'])

# ── 3. Generate prediction for the reference locus ───────────────────────────

# Reference locus: chr10:20.5–22.5 Mb = local bins 150–350
PATCH_START = 150
PATCH_SPAN  = 200   # 200 bins × 10 kb = 2 Mb

pred_band = np.array(jax_forward_generate(
    jnp.array(p_left [PATCH_START:PATCH_START + PATCH_SPAN], jnp.float32),
    jnp.array(p_right[PATCH_START:PATCH_START + PATCH_SPAN], jnp.float32),
    0.025,
    PATCH_SPAN,
))
obs_band = band[:PATCH_SPAN, PATCH_START:PATCH_START + PATCH_SPAN]

# ── 4. Normalize and build split contact map ─────────────────────────────────

def _log_eo(b):
    return np.array(normalize_expected_observed(jnp.asarray(b, jnp.float32)))

pred_sq  = flip_diag_row(_log_eo(pred_band))
obs_sq   = flip_diag_row(_log_eo(obs_band))
# Upper triangle = observation, lower triangle = prediction
combined = np.triu(obs_sq) + np.tril(pred_sq.T, k=-1)

# ── 5. Figure (L track on top, contact map, R track on right) ─────────────────

cmap = sns.color_palette('vlag', as_cmap=True)

patch_start_mb = GENOMIC_OFFSET_MB + PATCH_START * RESOLUTION / 1e6   # 20.5
patch_end_mb   = patch_start_mb + PATCH_SPAN * RESOLUTION / 1e6       # 22.5

# Bin centres in Mb — used for track coordinates (aligned with matshow extent)
coords = np.linspace(patch_start_mb, patch_end_mb, PATCH_SPAN, endpoint=False)

L_patch = p_left [PATCH_START:PATCH_START + PATCH_SPAN]
R_patch = p_right[PATCH_START:PATCH_START + PATCH_SPAN]

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(
    2, 2,
    hspace=0, wspace=0,
    height_ratios=[1, 5], width_ratios=[5, 1],
)
ax_main  = fig.add_subplot(gs[1, 0])
ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

# Contact map — extent sets real Mb coordinates on both axes
im = ax_main.matshow(
    combined, cmap=cmap, vmin=-2, vmax=2,
    extent=[patch_start_mb, patch_end_mb, patch_end_mb, patch_start_mb],
)
ax_main.xaxis.set_ticks_position('bottom')
ax_main.tick_params(labelsize=8)
ax_main.set_xlabel('chr10 (Mb)', fontsize=9)
ax_main.set_ylabel('chr10 (Mb)', fontsize=9)
ax_main.text(0.97, 0.97, 'obs',  fontsize=8, ha='right', va='top',
             transform=ax_main.transAxes, color='white')
ax_main.text(0.03, 0.03, 'pred', fontsize=8, ha='left',  va='bottom',
             transform=ax_main.transAxes, color='white')

# L track (top) — shares x-axis with contact map
ax_top.plot(coords, L_patch, lw=1.5, c='tab:blue')
ax_top.set_ylim(0, 1.05)
ax_top.set_yticks([0, 0.5, 1])
ax_top.set_yticklabels(['', '', '1'], fontsize=7)
ax_top.set_ylabel('L', fontsize=9)
ax_top.xaxis.set_visible(False)
ax_top.set_title('chr10:20.5–22.5 Mb  |  H1-hESC 10 kb', fontsize=9)

# R track (right) — shares y-axis with contact map
ax_right.plot(R_patch, coords, lw=1.5, c='tab:orange')
ax_right.set_xlim(0, 1.05)
ax_right.set_xticks([0, 0.5, 1])
ax_right.set_xticklabels(['', '', '1'], fontsize=7)
ax_right.set_xlabel('R', fontsize=9)
ax_right.xaxis.set_ticks_position('top')
ax_right.xaxis.set_label_position('top')
ax_right.yaxis.set_visible(False)

# Force exact alignment — gridspec panels can drift slightly due to tick-label rendering
fig.canvas.draw()
main_bbox  = ax_main.get_position()
top_bbox   = ax_top.get_position()
right_bbox = ax_right.get_position()
total_w    = right_bbox.x0 + right_bbox.width - main_bbox.x0
right_w    = total_w / 6   # width_ratios [5, 1] → R panel is 1/6 of total
ax_top.set_position([main_bbox.x0, top_bbox.y0, main_bbox.width, top_bbox.height])
ax_right.set_position([main_bbox.x0 + main_bbox.width, main_bbox.y0,
                        right_w, main_bbox.height])

# Colorbar in its own axes to the right of the R panel — does not steal from it
cbar_ax = fig.add_axes([
    main_bbox.x0 + main_bbox.width + right_w + 0.01,
    main_bbox.y0,
    0.025,
    main_bbox.height,
])
fig.colorbar(im, cax=cbar_ax, label='log(obs/exp)')

out_png = os.path.join(os.path.dirname(__file__), 'quick_start_patch.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'Saved: {out_png}')
