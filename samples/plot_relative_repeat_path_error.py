import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path

def plot_paths(ax, test_graph, run1, run2, path_matrix1, omit_start, omit_end, vmin, vmax):
    v_start1 = test_graph.get_vertex((run1, 0))
    x1, y1, z1, t1 = [], [], [], []
    vertices1 = list(TemporalIterator(v_start1))
    for i, (v, e) in enumerate(vertices1):
        if i < omit_start or i >= len(vertices1) - omit_end:
            continue
        x1.append(v.T_v_w.r_ba_ina()[0])
        y1.append(v.T_v_w.r_ba_ina()[1])
        z1.append(v.T_v_w.r_ba_ina()[2])
        t1.append(v.stamp / 1e9)

    v_start2 = test_graph.get_vertex((run2, 0))
    x2, y2, z2, t2, dist = [], [], [], [], []
    vertices2 = list(TemporalIterator(v_start2))
    for i, (v, e) in enumerate(vertices2):
        if i < omit_start or i >= len(vertices2) - omit_end:
            continue
        x2.append(v.T_v_w.r_ba_ina()[0])
        y2.append(v.T_v_w.r_ba_ina()[1])
        z2.append(v.T_v_w.r_ba_ina()[2])
        t2.append(v.stamp / 1e9)
        dist.append(vtr_path.signed_distance_to_path(v.T_v_w.r_ba_ina(), path_matrix1))

    c = [abs(v) for v in dist]
    ax.scatter(x1, y1, label=f"Repeat {run1}")
    sc = ax.scatter(x2, y2, label=f"Repeat {run2}", c=c, vmin=vmin, vmax=vmax)
    ax.axis('equal')
    ax.set_xlabel('x (m)', fontsize=18)
    ax.set_ylabel('y (m)', fontsize=18)
    ax.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5, zorder=-1)
    ax.tick_params(axis='both', which='major', labelsize=16)  # Increase font size of axis numbers
    return sc

if __name__ == '__main__':
    offline_graph_dirs = ["/home/desiree/ASRL/vtr3/temp/finalDome/graph", "/home/desiree/ASRL/vtr3/temp/finalUTIASParking/graph", "/home/desiree/ASRL/vtr3/temp/finalUTPParking/graph", "/home/desiree/ASRL/vtr3/temp/finalUTPSurvey/graph"]
    runs = [(2, 3), (1, 2), (1, 2), (1, 2)]
    omit_starts = [10, 60, 60, 60]  # Number of vertices to omit from the start for each graph
    omit_ends = [20, 60, 60, 60]    # Number of vertices to omit from the end for each graph

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    plt.rcParams.update({'font.size': 18})  # Set default font size

    all_distances = []

    for offline_graph_dir, (run1, run2), omit_start, omit_end in zip(offline_graph_dirs, runs, omit_starts, omit_ends):
        factory = Rosbag2GraphFactory(offline_graph_dir)
        test_graph = factory.buildGraph()
        g_utils.set_world_frame(test_graph, test_graph.root)
        path_matrix1 = vtr_path.path_to_matrix(test_graph, TemporalIterator(test_graph.get_vertex((run1, 0))))
        v_start2 = test_graph.get_vertex((run2, 0))
        vertices2 = list(TemporalIterator(v_start2))
        for i, (v, e) in enumerate(vertices2):
            if i < omit_start or i >= len(vertices2) - omit_end:
                continue
            all_distances.append(abs(vtr_path.signed_distance_to_path(v.T_v_w.r_ba_ina(), path_matrix1)))

    vmin, vmax = min(all_distances), max(all_distances)

    for i, (offline_graph_dir, (run1, run2), omit_start, omit_end) in enumerate(zip(offline_graph_dirs, runs, omit_starts, omit_ends)):
        factory = Rosbag2GraphFactory(offline_graph_dir)
        test_graph = factory.buildGraph()
        g_utils.set_world_frame(test_graph, test_graph.root)
        path_matrix1 = vtr_path.path_to_matrix(test_graph, TemporalIterator(test_graph.get_vertex((run1, 0))))
        sc = plot_paths(axs[i // 2, i % 2], test_graph, run1, run2, path_matrix1, omit_start, omit_end, vmin, vmax)

    cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Relative Lateral Error Between Repeats (m)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for colorbar
    plt.show()
