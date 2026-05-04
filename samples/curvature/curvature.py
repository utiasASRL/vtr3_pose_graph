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
# Compute |Δκ| for teach paths
# ============================
teach_office['c']  = np.abs(np.diff(teach_office['c']))
teach_urban['c']   = np.abs(np.diff(teach_urban['c']))
repeat_office['c'] = np.asarray(repeat_office['c'])
repeat_urban['c']  = np.asarray(repeat_urban['c'])

# ============================
# Shared colour limits across both environments
# ============================
vmin_curv = min(teach_office['c'].min(),  teach_urban['c'].min())
vmax_curv = max(teach_office['c'].max(),  teach_urban['c'].max())
vmin_err  = min(repeat_office['c'].min(), repeat_urban['c'].min())
vmax_err  = max(repeat_office['c'].max(), repeat_urban['c'].max())

# ============================
# Zoom regions
# ============================
office_zoom_xlim = (20, 70)
office_zoom_ylim = (-25, 45)

urban_zoom_xlim = (270, 320)
urban_zoom_ylim = (35, 105)

# ============================
# Peak |Δκ| annotation
# ============================
def _annotate_peak(ax, xs, ys, cs, zoom_xlim, zoom_ylim, text, color, dx_frac, dy_frac):
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()
    cs = np.asarray(cs).ravel()
    mask = (
        (xs >= zoom_xlim[0]) & (xs <= zoom_xlim[1]) &
        (ys >= zoom_ylim[0]) & (ys <= zoom_ylim[1])
    )
    if not mask.any():
        return
    idx = np.argmax(cs[mask])
    px = float(xs[mask][idx])
    py = float(ys[mask][idx])
    dx = (zoom_xlim[1] - zoom_xlim[0]) * dx_frac
    dy = (zoom_ylim[1] - zoom_ylim[0]) * dy_frac
    ax.annotate(
        text,
        xy=(px, py),
        xytext=(px - dx, py - dy),
        fontsize=12,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
    )

def annotate_peak_curv(ax, teach, zoom_xlim, zoom_ylim, text):
    _annotate_peak(ax, teach['x'][1:], teach['y'][1:], teach['c'],
                   zoom_xlim, zoom_ylim, text, "darkred", 0.4, 0.2)

def annotate_peak_err(ax, repeat, zoom_xlim, zoom_ylim, text):
    _annotate_peak(ax, repeat['x'], repeat['y'], repeat['c'],
                   zoom_xlim, zoom_ylim, text, "darkblue", 0.5, -0.1)

# ============================
# Figure: 1×2 zoomed panels + 2 colorbars
# ============================
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(
    1, 5,
    width_ratios=[0.8,0.8, 0.04, 0.15, 0.04],
    wspace=0.1,
)
ax_off   = fig.add_subplot(gs[0, 0])
ax_urb   = fig.add_subplot(gs[0, 1])
cax_curv = fig.add_subplot(gs[0, 2])
cax_err  = fig.add_subplot(gs[0, 4])

# ============================
# Plotting helper
# ============================
def plot_traj(ax, teach, repeat, zoom_xlim, zoom_ylim):
    sc_teach = ax.scatter(
        teach['x'][1:], teach['y'][1:],
        c=teach['c'], s=50, cmap="Reds",
        vmin=vmin_curv, vmax=vmax_curv,
        zorder=1, linewidths=0
    )
    sc_err = ax.scatter(
        repeat['x'], repeat['y'],
        c=repeat['c'], s=8, cmap="viridis",
        vmin=vmin_err, vmax=vmax_err,
        zorder=2, linewidths=0
    )
    ax.set_xlim(zoom_xlim)
    ax.set_ylim(zoom_ylim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.4, color="grey", alpha=0.5, zorder=0)
    return sc_teach, sc_err

# ============================
# Plot
# ============================
sc_teach, sc_err = plot_traj(ax_off, teach_office, repeat_office,
                              office_zoom_xlim, office_zoom_ylim)
plot_traj(ax_urb, teach_urban, repeat_urban,
          urban_zoom_xlim, urban_zoom_ylim)

# ============================
# Annotations
# ============================
annotate_peak_curv(ax_off, teach_office, office_zoom_xlim, office_zoom_ylim, text=r"Max $|\Delta$ Curvature|")
annotate_peak_curv(ax_urb, teach_urban,  urban_zoom_xlim,  urban_zoom_ylim,  text=r"Max $|\Delta$ Curvature|")
annotate_peak_err(ax_off, repeat_office, office_zoom_xlim, office_zoom_ylim, text="Max Tracking Error")
annotate_peak_err(ax_urb, repeat_urban,  urban_zoom_xlim,  urban_zoom_ylim,  text="Max Tracking Error")

# ============================
# Panel labels and titles
# ============================
for ax, label in zip([ax_off, ax_urb], ["(a)", "(b)"]):
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

ax_off.set_title("Urban-R (Zoomed-In)", fontsize=12)
ax_urb.set_title("Structured-R (Zoomed-In)", fontsize=12)

# ============================
# Colorbars
# ============================
cb_curv = fig.colorbar(sc_teach, cax=cax_curv)
cb_curv.set_label(r"Teach Path Rate of Change of Curvature (m$^{-1}$)", fontsize=12)
cb_curv.ax.tick_params(labelsize=8)

cb_err = fig.colorbar(sc_err, cax=cax_err)
cb_err.set_label("Repeat Path Lateral Error (m)", fontsize=12)
cb_err.ax.tick_params(labelsize=8)

plt.savefig('curvature.png', dpi=200, bbox_inches='tight')
plt.show()
