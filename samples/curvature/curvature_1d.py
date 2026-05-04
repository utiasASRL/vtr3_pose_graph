import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import pickle as pkl
import os

# ============================
# Load data
# ============================
pkl_dir = "curvature/"

for fname in os.listdir(pkl_dir):
    if fname.endswith(".pkl"):
        with open(os.path.join(pkl_dir, fname), "rb") as f:
            data = pkl.load(f)
        if fname.endswith("office_t.pkl"):
            teach_office = data
        if fname.endswith("office_r.pkl"):
            repeat_office = data
        if fname.endswith("urban_t.pkl"):
            teach_urban = data
        if fname.endswith("urban_r.pkl"):
            repeat_urban = data

# ============================
# Helpers
# ============================
def cum_path_length(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.concatenate([[0.0], np.cumsum(ds)])

# ============================
# Compute |Δκ| and path lengths
# ============================
teach_office_kappa  = np.asarray(teach_office['c']).ravel()
teach_urban_kappa   = np.asarray(teach_urban['c']).ravel()
teach_office_dkappa = np.abs(np.diff(teach_office_kappa))
teach_urban_dkappa  = np.abs(np.diff(teach_urban_kappa))

repeat_office_err = np.asarray(repeat_office['c']).ravel()
repeat_urban_err  = np.asarray(repeat_urban['c']).ravel()

teach_office_s  = cum_path_length(teach_office['x'],  teach_office['y'])
teach_urban_s   = cum_path_length(teach_urban['x'],   teach_urban['y'])
repeat_office_s = cum_path_length(repeat_office['x'], repeat_office['y'])
repeat_urban_s  = cum_path_length(repeat_urban['x'],  repeat_urban['y'])

# |Δκ| is one element shorter than the teach path; align to s[1:]
teach_office_s_dk = teach_office_s[1:]
teach_urban_s_dk  = teach_urban_s[1:]

# ============================
# Figure: 2 rows × 2 cols, shared x within each column
# ============================
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 2, hspace=0.08, wspace=0.30)

ax_curv_off  = fig.add_subplot(gs[0, 0])
ax_err_off   = fig.add_subplot(gs[1, 0], sharex=ax_curv_off)
ax_curv_urb  = fig.add_subplot(gs[0, 1])
ax_err_urb   = fig.add_subplot(gs[1, 1], sharex=ax_curv_urb)

COLOR_CURV = "darkred"
COLOR_ERR  = "#1f77b4"  # default matplotlib blue

# ============================
# Office column
# ============================
ax_curv_off.plot(teach_office_s_dk, teach_office_dkappa,
                 color=COLOR_CURV, lw=0.9, alpha=0.85)
ax_curv_off.fill_between(teach_office_s_dk, teach_office_dkappa,
                          color=COLOR_CURV, alpha=0.15)

ax_err_off.plot(repeat_office_s, repeat_office_err,
                color=COLOR_ERR, lw=0.9, alpha=0.85)
ax_err_off.fill_between(repeat_office_s, repeat_office_err,
                         color=COLOR_ERR, alpha=0.15)

# ============================
# Urban column
# ============================
ax_curv_urb.plot(teach_urban_s_dk, teach_urban_dkappa,
                 color=COLOR_CURV, lw=0.9, alpha=0.85)
ax_curv_urb.fill_between(teach_urban_s_dk, teach_urban_dkappa,
                          color=COLOR_CURV, alpha=0.15)

ax_err_urb.plot(repeat_urban_s, repeat_urban_err,
                color=COLOR_ERR, lw=0.9, alpha=0.85)
ax_err_urb.fill_between(repeat_urban_s, repeat_urban_err,
                         color=COLOR_ERR, alpha=0.15)

# ============================
# Axis labels and formatting
# ============================
for ax in [ax_curv_off, ax_curv_urb, ax_err_off, ax_err_urb]:
    ax.grid(True, linewidth=0.4, color="grey", alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

# Hide x tick labels on top row (shared axis)
plt.setp(ax_curv_off.get_xticklabels(), visible=False)
plt.setp(ax_curv_urb.get_xticklabels(), visible=False)

ax_curv_off.set_ylabel(r"$|\Delta\kappa|$ (m$^{-1}$ node$^{-1}$)", fontsize=9)
ax_err_off.set_ylabel("Lateral error (m)", fontsize=9)
ax_err_off.set_xlabel("Path length (m)", fontsize=9)
ax_err_urb.set_xlabel("Path length (m)", fontsize=9)

# ============================
# Column titles and panel labels
# ============================
ax_curv_off.set_title("Indoor (Office)", fontsize=10)
ax_curv_urb.set_title("Outdoor (Urban)", fontsize=10)

for ax, label in zip(
    [ax_curv_off, ax_curv_urb, ax_err_off, ax_err_urb],
    ["(a)", "(b)", "(c)", "(d)"]
):
    ax.text(0.01, 0.97, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

plt.savefig('curvature_1d.png', dpi=200, bbox_inches='tight')
plt.show()
