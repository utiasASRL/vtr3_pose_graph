import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

import numpy as np
import pickle as pkl
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator, TemporalIterator, BreadthFirstSearchIterator, DepthFirstSearchIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

import pylgmath

import pdb

def curvature_2d(points):
    """
    Compute curvature along a 2D path using 3-point circle method.

    points: (N, 2) array of [x, y]
    returns: (N,) array of curvature values
    """
    points = np.asarray(points)
    N = len(points)
    kappa = np.zeros(N)

    for i in range(1, N - 1):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]

        a = p1 - p0
        b = p2 - p1
        c = p2 - p0

        # Triangle area * 2 (cross product magnitude)
        cross = np.abs(a[0]*b[1] - a[1]*b[0])

        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        lc = np.linalg.norm(c)

        # Avoid division by zero (collinear or duplicate points)
        if la * lb * lc > 0:
            kappa[i] = 2 * cross / (la * lb * lc)
        else:
            kappa[i] = 0.0

    # Optional: copy interior values to endpoints
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]

    return kappa

def compose_with_cov(
    T_prev: pylgmath.TransformationWithCovariance,
    T_edge: pylgmath.TransformationWithCovariance,
) -> pylgmath.TransformationWithCovariance:
    """
    Compose two TransformationWithCovariance and propagate covariance.
    cov_k = Ad(T_edge^-1) @ cov_{k-1} @ Ad(T_edge^-1).T + cov_edge
    """
    T_composed = pylgmath.TransformationWithCovariance(
        T_ba=(T_prev @ T_edge).matrix()
    )
    Ad_inv = T_edge.inverse().adjoint()
    cov_composed = Ad_inv @ T_prev.cov() @ Ad_inv.T + T_edge.cov()
    T_composed.set_covariance(cov_composed)
    # T_composed.set_covariance(T_edge.cov()) #if we want to see the raw value
    return T_composed

custom_colors = [
  "#3cb44b", #  0 - mr_green
  "#911eb4", #  1 - prof_plum
  "#ffe119", #  2 - col_mustard
  "#4363d8", #  3 - mrs_peacock
  "#e6194b", #  4 - red
  "#f58231", #  5 - orange
  "#bfef45", #  6 - lime
  "#42d4f4", #  7 - cyan
  "#f032e6", #  8 - magenta
  "#fabebe", #  9 - pink
  "#ffd8b1", # 10 - apricot
  "#fffac8", # 11 - cream
  "#aaffc3", # 12 - mint
  "#a9a9a9", # 13 - grey
  "#ffffff", # 14 - white
  "#000000", # 15 - black
]

# Create the custom colormap
custom_cmap = mcolors.ListedColormap(custom_colors)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Verify Point Cloud',
                        description = 'Plots point cloud to verify alignment')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"))      # option that takes a value
    args = parser.parse_args()

    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")
    # root = next(iter(test_graph._vertices.values()))
    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.root

    x = []
    y = []
    t = []
    c = []
    T = []

    path_len=0
    teach_count = 0
    count = 0

    for v, e in BreadthFirstSearchIterator(v_start):
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        t.append(v.stamp / 1e9)
        T_edge = pylgmath.TransformationWithCovariance(
            transformation=e.T, covariance=e.cov
        )
        if count == 0:
            T_odom = [pylgmath.TransformationWithCovariance(init_cov_to_zero=True)]
        else:
            T_odom.append(compose_with_cov(T_odom[-1], T_edge))

        if e.mode == 1 and (e.from_id < e.to_id):
            teach_count += 1
        count += 1
        # c.append(e.mode)

        c.append((v.id >> 60) & 0xF) # color based on rid
            
        path_len += np.linalg.norm(e.T.r_ba_ina())

    covs = [T.cov() for T in T_odom[1:]]  # skip root, cov is zero

    EVERY_N = 1000000000  # tune based on graph density
    fig, ax = plt.subplots()

    for i, cov in enumerate(covs[::EVERY_N]):
        xi = x[i * EVERY_N + 1]  # +1 because covs skips root
        yi = y[i * EVERY_N + 1]

        # x-y translation block — indices [0,1] assuming [rho; phi] ordering
        # cov_xy = cov[np.ix_([0, 1], [0, 1])]
        cov_xy = cov[0:2,0:2]

        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        eigvals = np.maximum(eigvals, 0)  # numerical safety for near-zero values

        width  = 3 * 2 * np.sqrt(eigvals[0])  # 3-sigma
        height = 3 * 2 * np.sqrt(eigvals[1])
        angle  = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        ax.add_patch(Ellipse(
            xy=(xi, yi), width=width, height=height, angle=angle,
            edgecolor='tomato', facecolor='tomato', alpha=0.1, linewidth=0.6,
        ))

    c_map = mcolors.ListedColormap(custom_colors)
    print(f"path length : {path_len}")
    print(f"num taught vertices : {teach_count}, num total vertices : {count}")    
    sc = ax.scatter(x, y, label="Teach", c=c, cmap=c_map, vmin=-0.5, vmax=15.5)
    # ax.set_title()
    id = args.graph[-8]
    ax.set_title(
        f"vtr_pose_graph", 
        fontsize=14, 
        fontweight='bold', 
        pad=15,
        color=c_map(int(id))
    )
    cbar = plt.colorbar(sc, ticks=range(16))
    # plt.colorbar(label='Autonomous (0), Manual (1), Unknown (2)', ax=None)
    cbar.set_label('Run ID (rid)')

    plt.axis('equal')
    plt.show()
