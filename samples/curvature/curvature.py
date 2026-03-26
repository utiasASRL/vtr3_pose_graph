import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import pickle as pkl
import os
from matplotlib.patches import Rectangle

# ============================
# Helpers
# ============================
def draw_zoom_box(ax, xlim, ylim, **kwargs):
    rect = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        fill=False,
        linewidth=2,
        **kwargs
    )
    ax.add_patch(rect)

def connect_axes(fig, ax_from, ax_to):
    bbox_from = ax_from.get_position()
    bbox_to   = ax_to.get_position()

    start = (bbox_from.x1, bbox_from.y0 + bbox_from.height / 2)
    end   = (bbox_to.x0,   bbox_to.y0 + bbox_to.height / 2)

    ax_from.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
        xycoords="figure fraction",
        textcoords="figure fraction",
    )

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
# Normalize curvature & error
# ============================
teach_office['c']  = np.abs(np.diff(teach_office['c']))
teach_urban['c']   = np.abs(np.diff(teach_urban['c']))
repeat_office['c'] = np.asarray(repeat_office['c'])
repeat_urban['c']  = np.asarray(repeat_urban['c'])

teach_office['c']  /= teach_office['c'].max()
teach_urban['c']   /= teach_urban['c'].max()
repeat_office['c'] /= repeat_office['c'].max()
repeat_urban['c']  /= repeat_urban['c'].max()

# ============================
# Zoom regions (defined ONCE)
# ============================
office_zoom_xlim = (35, 75)
office_zoom_ylim = (-10, 30)

urban_zoom_xlim = (270, 330)
urban_zoom_ylim = (40, 100)

# ============================
# Figure layout
# ============================
fig = plt.figure(figsize=(12, 9))
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(
3, 3,
width_ratios=[1, 1, 0.05],
height_ratios=[1, 1, 0.06],
wspace=0.25,
hspace=0.30
)
ax_off_full = fig.add_subplot(gs[0, 0])
ax_urb_full = fig.add_subplot(gs[0, 1])
ax_off_zoom = fig.add_subplot(gs[1, 0])
ax_urb_zoom = fig.add_subplot(gs[1, 1])
cax_err = fig.add_subplot(gs[0:2, 2])
# cax_curv  = fig.add_subplot(gs[2, 0:2])
# ============================
# Plotting helper
# ============================
def plot_traj(ax, teach, repeat, title=None, zoom=None, square_box=False):
    # ax.grid(zorder=10)

    sc_teach = ax.scatter(
        teach['x'][1:], teach['y'][1:],
        c=teach['c'], s=80, cmap="Reds"
    )
    sc_err = ax.scatter(
        repeat['x'], repeat['y'],
        c=repeat['c'], s=10, cmap="viridis"
    )

    if zoom is not None:
        ax.set_xlim(zoom[0])
        ax.set_ylim(zoom[1])
        ax.set_aspect("equal", "box")
    else:
        ax.set_aspect("equal", adjustable="datalim")
        if square_box:
            ax.set_box_aspect(1)

    if title:
        ax.set_title(title)
    # ax.legend()

    return sc_teach, sc_err

# ============================
# Global views (square boxes)
# ============================
sc_teach, sc_err = plot_traj(
    ax_off_full,
    teach_office,
    repeat_office,
    # title="Office (Global View)",
    square_box=True
)

plot_traj(
    ax_urb_full,
    teach_urban,
    repeat_urban,
    # title="Urban (Global View)",
    square_box=True
)

# ============================
# Zoomed views
# ============================
plot_traj(
    ax_off_zoom,
    teach_office,
    repeat_office,
    # title="Office (Zoomed)",
    zoom=(office_zoom_xlim, office_zoom_ylim)
)

plot_traj(
    ax_urb_zoom,
    teach_urban,
    repeat_urban,
    # title="Urban (Zoomed)",
    zoom=(urban_zoom_xlim, urban_zoom_ylim)
)

# ============================
# Draw zoom boxes on globals
# ============================
draw_zoom_box(
    ax_off_full,
    office_zoom_xlim,
    office_zoom_ylim,
    edgecolor="grey"
)

draw_zoom_box(
    ax_urb_full,
    urban_zoom_xlim,
    urban_zoom_ylim,
    edgecolor="grey"
)

# ============================
# Connect global → zoom
# ============================
# connect_axes(fig, ax_off_full, ax_off_zoom)
# connect_axes(fig, ax_urb_full, ax_urb_zoom)

# ============================
# Colorbars
# ============================
cb_err = fig.colorbar(sc_err, cax=cax_err)
cb_err.set_label("Normalized Repeat Path Tracking Error")

# cb_err = fig.colorbar(sc_err, cax=cax_err, orientation="horizontal")
# cb_err.set_label("Normalized Repeat Path Tracking Error")
# plt.grid()
plt.tight_layout()
plt.savefig('curvature.png')
plt.show()