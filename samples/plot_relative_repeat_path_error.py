import os
import matplotlib.pyplot as plt
import numpy as np
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import TemporalIterator
import math
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
    #aligned_distances = dist[:len(t2)]
    #aligned_t = t2[:len(aligned_distances)]
    #rmse = np.sqrt(np.trapz(np.array(aligned_distances)**2, aligned_t) / (aligned_t[-1] - aligned_t[0]))
    #max_error = max(abs(v) for v in aligned_distances)

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
     
     offline_graph_dirs = [
            "/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/dome/graph",
            "/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/parking/graph",
            "/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/bigpath/graph",
            "/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/grassy/graph"
     ]
     runs = [
            (1, 3, 4, 5, 6),
            (4, 7, 8, 9, 10),   
            (1, 2, 3, 4, 5),
            (1, 2, 3, 5, 6)
     ]

     omit_starts = [0, 0, 0, 0]  # Number of vertices to omit from the start for each graph
     omit_ends = [2, 2, 2, 2]    # Number of vertices to omit from the end for each graph

     # Determine grid layout dynamically based on the number of graphs
     n_graphs = len(offline_graph_dirs)
     n_cols = 4  # modify as needed
     n_rows = math.ceil(n_graphs / n_cols)
     fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 18)) #15

     # Make axs a flat list for easy indexing.
     if n_graphs == 1:
            axs = [axs]
     else:
            axs = np.array(axs).flatten()

     plt.rcParams.update({'font.size': 18})  # Set default font size

     # First pass: compute global min and max error across all comparison runs for color scaling
     all_distances = []
     for offline_graph_dir, run_set, omit_start, omit_end in zip(offline_graph_dirs, runs, omit_starts, omit_ends):
            factory = Rosbag2GraphFactory(offline_graph_dir)
            test_graph = factory.buildGraph()
            g_utils.set_world_frame(test_graph, test_graph.root)
            # Compute the reference path from the first run in the tuple.
            base_path_matrix = vtr_path.path_to_matrix(
                    test_graph, TemporalIterator(test_graph.get_vertex((run_set[0], 0)))
            )
            # Loop over all comparison runs (all runs after the first)
            for run in run_set[1:]:
                    v_start = test_graph.get_vertex((run, 0))
                    vertices = list(TemporalIterator(v_start))
                    for i, (v, e) in enumerate(vertices):
                           if i < omit_start or i >= len(vertices) - omit_end:
                                   continue
                           d = abs(vtr_path.signed_distance_to_path(v.T_v_w.r_ba_ina(), base_path_matrix))
                           all_distances.append(d)
     vmin, vmax = min(all_distances), max(all_distances)

# Second pass: for each graph, plot the paths and cumulative error metric
# We also produce an error plot for each graph in a separate window.
sc = None  # placeholder for color scatter handle for use in colorbar
for i, (offline_graph_dir, run_set, omit_start, omit_end) in enumerate(zip(offline_graph_dirs, runs, omit_starts, omit_ends)):
         factory = Rosbag2GraphFactory(offline_graph_dir)
         test_graph = factory.buildGraph()
         g_utils.set_world_frame(test_graph, test_graph.root)
         ax = axs[i]

         # Prepare the base (reference) run plot.
         base_run = run_set[0]
         base_vertex = test_graph.get_vertex((base_run, 0))
         base_vertices = list(TemporalIterator(base_vertex))
         x_base, y_base = [], []
         for j, (v, e) in enumerate(base_vertices):
                     if j < omit_start or j >= len(base_vertices) - omit_end:
                              continue
                     pt = v.T_v_w.r_ba_ina()
                     x_base.append(pt[0])
                     y_base.append(pt[1])
         ax.scatter(x_base, y_base, color='black')

         # Compute the reference path matrix.
         base_path_matrix = vtr_path.path_to_matrix(
                     test_graph, TemporalIterator(test_graph.get_vertex((base_run, 0)))
         )

         cumulative_distances = []  # collect errors from all comparison runs

         # Create a separate figure for cumulative error vs time for this graph.
         fig_error, ax_error = plt.subplots(figsize=(10, 5))
         for run in run_set[1:]:
                     vtx = test_graph.get_vertex((run, 0))
                     vertices = list(TemporalIterator(vtx))
                     x_run, y_run, t_run, dist = [], [], [], []
                     for j, (v, e) in enumerate(vertices):
                              if j < omit_start or j >= len(vertices) - omit_end:
                                      continue
                              pt = v.T_v_w.r_ba_ina()
                              x_run.append(pt[0])
                              y_run.append(pt[1])
                              t_run.append(v.stamp / 1e9)
                              d = vtr_path.signed_distance_to_path(pt, base_path_matrix)
                              dist.append(d)
                     cumulative_distances.extend(dist)
                     # Compute error metrics for this run.
                     if dist:
                            rmse_run = np.sqrt(np.mean(np.array(dist) ** 2))
                            max_error_run = max(abs(val) for val in dist)
                     else:
                            rmse_run, max_error_run = 0, 0
                     # Plot error curve for this run.
                     ax_error.plot(t_run, dist)
                     # Scatter positions on the map colored by error magnitude.
                     sc = ax.scatter(x_run, y_run, c=[abs(val) for val in dist],
                                           vmin=vmin, vmax=vmax, cmap='viridis')
         # Compute cumulative error metrics over all comparison runs.
         if cumulative_distances:
                     cumulative_rmse = np.sqrt(np.mean(np.array(cumulative_distances) ** 2))
                     cumulative_max = max(abs(val) for val in cumulative_distances)
         else:
                     cumulative_rmse, cumulative_max = 0, 0

         print(f"Graph: {offline_graph_dir} - Cumulative RMSE: {cumulative_rmse:.3f}m, Cumulative Max Error: {cumulative_max:.3f}m")

         ax_error.set_ylabel("Path Tracking Error (m)")
         ax_error.set_xlabel("Time (s)")
         ax_error.set_title(f"Cumulative Error: RMSE {cumulative_rmse:.3f}m, Max {cumulative_max:.3f}m")
         ax_error.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5, zorder=-1)
         # Removed legend for the error plot
         
         # Configure the main subplot.
         ax.set_box_aspect(1)
         ax.axis('equal')
         # Removed individual x and y labels for subplots
         ax.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5, zorder=-1)
         ax.tick_params(axis='both', which='major', labelsize=16)
         # Removed legend for the subplot

# Turn off any unused subplots.
for idx in range(n_graphs, len(axs)):
         axs[idx].axis('off')

# Create a colorbar from the last scatter plot.
if sc is not None:
         cbar = fig.colorbar(sc, ax=axs.tolist(), orientation='vertical', fraction=0.02, pad=0.04)
         cbar.set_label("Relative Lateral Error Between Repeats (m)", fontsize=18, labelpad=20)
         cbar.ax.tick_params(labelsize=16)

# Set common x and y axis labels for all subplots.
fig.supxlabel("x (m)", fontsize=18)
y_label = fig.supylabel("y (m)", fontsize=18)
# Shift the y-label closer to the subplots by adjusting its x-position.
y_label.set_x(0.06)
# Set the overall figure title.
fig.suptitle("VirLTR (Pix4D) Relative Repeat Deviation", fontsize=20)

#plt.tight_layout(pad=0.25)
plt.show()
