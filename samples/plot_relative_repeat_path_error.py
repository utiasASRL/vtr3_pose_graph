'''
import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path


def plot_paths(ax, test_graph, run1, run2, path_matrix1, omit_start, omit_end):
    path_len1, path_len2 = 0, 0
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
        path_len1 += np.linalg.norm(e.T.r_ba_ina()) 

    print(f"Path {run1} was {path_len1:.3f}m long") 

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
        path_len2 += np.linalg.norm(e.T.r_ba_ina()) 
        
    print(f"Path {run2} was {path_len2:.3f}m long") 

    # Compute RMSE and max error
    aligned_distances = dist[:len(t2)]
    aligned_t = t2[:len(aligned_distances)]
    rmse = np.sqrt(np.trapz(np.array(aligned_distances)**2, aligned_t) / (aligned_t[-1] - aligned_t[0]))
    max_error = max(abs(v) for v in aligned_distances)

    # Plot path tracking error
    fig, ax_error = plt.subplots()
    ax_error.plot(t2, dist, label=f"RMSE: {rmse:.3f}m, Max Error: {max_error:.3f}m between Repeats {run1} and {run2}")
    ax_error.set_ylabel("Path Tracking Error (m)")
    ax_error.set_xlabel("Time (s)")
    ax_error.set_title("Path Tracking Error")
    ax_error.legend()
    ax_error.set_xlim(min(t2), max(t2))  # Set x-axis limits to the range of t2
    ax_error.set_ylim(min(dist) - 0.1, max(dist) + 0.1)  # Set y-axis limits with a small margin

    # Plot paths for the current graph
    c = [abs(v) for v in dist]
    plt.scatter(x1, y1, label=f"Repeat {run1}")
    plt.scatter(x2, y2, label=f"Repeat {run2} (Max Error: {max_error:.3f}m between Repeats)", c=c)
    plt.axis('equal')
    plt.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5, zorder=-1)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend()

    c = [abs(v) for v in dist]
    ax.scatter(x1, y1, label=f"Repeat {run1}")
    sc = ax.scatter(x2, y2, label=f"Repeat {run2}", c=c)
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

   for i, (offline_graph_dir, (run1, run2), omit_start, omit_end) in enumerate(zip(offline_graph_dirs, runs, omit_starts, omit_ends)):
       factory = Rosbag2GraphFactory(offline_graph_dir)
       test_graph = factory.buildGraph()
       g_utils.set_world_frame(test_graph, test_graph.root)
       path_matrix1 = vtr_path.path_to_matrix(test_graph, TemporalIterator(test_graph.get_vertex((run1, 0))))
       sc = plot_paths(axs[i // 2, i % 2], test_graph, run1, run2, path_matrix1, omit_start, omit_end)

   cbar = fig.colorbar(sc, ax=axs.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
   cbar.set_label("Relative Lateral Error Between Repeats (m)", fontsize=18)
   cbar.ax.tick_params(labelsize=16)

   plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for colorbar
   plt.show()
'''

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


    # Compute RMSE and max error
    rmse = np.sqrt(np.mean(np.array(dist) ** 2))
    max_error = max(abs(v) for v in dist)
    print(f"RMSE for runs {run1} and {run2}: {rmse:.3f}m, Max Error: {max_error:.3f}m")

    # Compute RMSE and max error
    aligned_distances = dist[:len(t2)]
    aligned_t = t2[:len(aligned_distances)]
    rmse = np.sqrt(np.trapz(np.array(aligned_distances)**2, aligned_t) / (aligned_t[-1] - aligned_t[0]))
    max_error = max(abs(v) for v in aligned_distances)

    # Plot path tracking error
    fig, ax_error = plt.subplots(figsize=(10, 5))  # Set a rectangular aspect ratio
    ax_error.plot(t2, dist, label=f"RMSE: {rmse:.3f}m, Max Error: {max_error:.3f}m between Repeats {run1} and {run2}")
    ax_error.set_ylabel("Path Tracking Error (m)")
    ax_error.set_xlabel("Time (s)")
    ax_error.set_title("Path Tracking Error")
    ax_error.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5, zorder=-1)
    ax_error.legend()
    ax_error.set_xlim(min(t2), max(t2))  # Set x-axis limits to the range of t2
    ax_error.set_ylim(min(dist) - 0.1, max(dist) + 0.1)  # Set y-axis limits with a small margin

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
