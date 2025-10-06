import os
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import argparse
import time
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
from vtr_utils.plot_utils import extract_map_from_vertex

def collect_points_with_progress(graph, vertices):
    """Extract submap points and the normalized cumulative distance (0..1)
    for only those vertices that actually yield a submap."""
    points = []
    s_vals = []
    cum = 0.0
    for idx, (v, e) in enumerate(vertices):
        # advance arc length if we have an edge transform
        if e is not None:
            try:
                step = np.linalg.norm(e.T.r_ba_ina())
                if step is not None:
                    cum += float(step)
            except Exception:
                pass

        new_points, _ = extract_map_from_vertex(graph, v)
        if new_points is not None and new_points.size > 0:
            points.append(new_points)
            s_vals.append(cum)

    s_vals = np.asarray(s_vals, dtype=float)
    if s_vals.size and s_vals[-1] > 0:
        s_vals /= s_vals[-1]  # normalize to 0..1
    elif s_vals.size:
        s_vals[:] = 0.0
    return points, s_vals


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='Plot Teach and Repeat Submaps',
                        description='Plots submaps from teach and repeat overlaid. Also calculates RMS error')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="Path to the pose graph folder")
    parser.add_argument('-r', '--run', type=int, help="Select a repeat run.")
    args = parser.parse_args()

    # Build the graph and set the world frame
    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)
    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")
    g_utils.set_world_frame(test_graph, test_graph.root)

    # Compare signed distances
    v_start_teach = test_graph.root
    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start_teach))

    # Get teach path vertices (skipping first and last few)
    x, y, z, t = [], [], [], []
    vertices = list(PriviledgedIterator(v_start_teach))
    for i, (v, e) in enumerate(vertices):
        #if i < 15 or i >= len(vertices) - 15:
        pass
            #continue
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        z.append(v.T_v_w.r_ba_ina()[2])
        t.append(v.stamp / 1e9)

    plt.figure(0)
    plt.scatter(x, y, label="Teach")
    plt.axis('equal')
    plt.figure(1)
    plt.scatter(x, z, label="Teach")

    # Process repeat run for signed distance comparison
    v_start_repeat = test_graph.get_vertex((args.run, 0))
    x, y, z, t, dist = [], [], [], [], []
    path_len = 0
    vertices = list(TemporalIterator(v_start_repeat))
    for i, (v, e) in enumerate(vertices):
        #if i < 50 or i >= len(vertices) - 50:
        pass
            #continue
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        z.append(v.T_v_w.r_ba_ina()[2])
        t.append(v.stamp / 1e9)
        d = vtr_path.signed_distance_to_path(v.T_v_w.r_ba_ina(), path_matrix)
        dist.append(d)
        path_len += np.linalg.norm(e.T.r_ba_ina())
    
    max_error = max(abs(v) for v in dist)
    print(f"Path {args.run} was {path_len:.3f}m long")
    if len(t) > 2:
        c = [abs(v) for v in dist]
        plt.figure(0)
        plt.scatter(x, y, label=f"Repeat {args.run} (Max Error: {max_error:.3f}m)", c=c)
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.colorbar(label="Lateral Error (m)")
        plt.legend()
        plt.figure(1)
        plt.scatter(x, z, label=f"Repeat {args.run} (Max Error: {max_error:.3f}m)", c=c)
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.colorbar(label="Lateral Error (m)")
        plt.legend()
        plt.show()
'''
    # Extract and visualize submaps from both teach and repeat runs
    first = True
    paused = False

    def toggle(vis):
        global paused
        paused = not paused
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(' '), toggle)
    vis.create_window()

    teach_pcd = o3d.geometry.PointCloud()
    repeat_pcd = o3d.geometry.PointCloud()
    vis.poll_events()
    vis.update_renderer()

    # # Gather teach submaps by iterating over all teach runs
    # teach_points = []
    # for i in range(test_graph.major_id + 1):
    #     vertices = list(TemporalIterator(v_start_teach))
    #     # Optionally filter out the very first/last vertices
    #     vertices_to_plot = vertices[:-10] if len(vertices) > 10 else vertices
    #     for v, e in vertices_to_plot:
    #         new_points, map_ptr = extract_map_from_vertex(test_graph, v)
    #         if new_points is not None and new_points.size > 0:
    #             teach_points.append(new_points)

    # Gather teach submaps from the privileged (teach) chain once
    teach_points = []
    teach_vertices = list(PriviledgedIterator(v_start_teach))  # <- privileged teach path
    teach_vertices_to_plot = teach_vertices[:-10] if len(teach_vertices) > 10 else teach_vertices
    for v, e in teach_vertices_to_plot:
        new_points, _ = extract_map_from_vertex(test_graph, v)
        if new_points is not None and new_points.size > 0:
            teach_points.append(new_points)

    # Gather repeat submaps (for the selected repeat run)
    repeat_points = []
    vertices = list(TemporalIterator(v_start_repeat))
    vertices_to_plot = vertices[:-10] if len(vertices) > 10 else vertices
    for v, e in vertices_to_plot:
        new_points, map_ptr = extract_map_from_vertex(test_graph, v)
        if new_points is not None and new_points.size > 0:
            repeat_points.append(new_points)

    # Now loop through the maximum length of the two lists
    max_len = max(len(teach_points), len(repeat_points))
    for i in range(max_len):
        # Update teach submap if available
        if i < len(teach_points):
            teach_pcd.points = o3d.utility.Vector3dVector(teach_points[i].T)
            teach_pcd.paint_uniform_color((1.0, 0.0, 0.0))  # Red for teach
            if first:
                vis.add_geometry(teach_pcd)
            else:
                vis.update_geometry(teach_pcd)

        if i < len(repeat_points):
            repeat_pcd.points = o3d.utility.Vector3dVector(repeat_points[i].T)
            repeat_pcd.paint_uniform_color((0.0, 1.0, 0.0))  # Green for repeat
            if first:
                vis.add_geometry(repeat_pcd)
            else:
                vis.update_geometry(repeat_pcd)

        print("Plotting submaps (red for teach, green for repeat)...")
        t0 = time.time()

        while time.time() - t0 < 0.1 or paused:
            vis.poll_events()
            vis.update_renderer()

        first = False

    vis.run()
    vis.destroy_window()

    
'''

# --- Extract and visualize submaps from both teach and repeat runs (ARC-LENGTH SYNC) ---
first = True
paused = False

def toggle(vis):
    global paused
    paused = not paused
    return False

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.register_key_callback(ord(' '), toggle)
vis.create_window()
teach_pcd = o3d.geometry.PointCloud()
repeat_pcd = o3d.geometry.PointCloud()
vis.poll_events()
vis.update_renderer()

# Use the privileged chain for TEACH, temporal for REPEAT
teach_vertices = list(PriviledgedIterator(v_start_teach))
repeat_vertices = list(TemporalIterator(v_start_repeat))

# Optionally trim ends
teach_vertices_to_plot = teach_vertices[:-10] if len(teach_vertices) > 10 else teach_vertices
repeat_vertices_to_plot = repeat_vertices[:-10] if len(repeat_vertices) > 10 else repeat_vertices

# Collect submaps + their cumulative-distance progress (normalized 0..1)
teach_points, s_teach = collect_points_with_progress(test_graph, teach_vertices_to_plot)
repeat_points, s_repeat = collect_points_with_progress(test_graph, repeat_vertices_to_plot)

if not teach_points:
    print("Warning: no teach submaps extracted.")
if not repeat_points:
    print("Warning: no repeat submaps extracted.")

max_len = max(len(teach_points), len(repeat_points))
if max_len == 0:
    vis.destroy_window()
    raise SystemExit("No submaps to show.")

# Build a common progress timeline and map to nearest index in each sequence
progress = np.linspace(0.0, 1.0, max_len) if max_len > 1 else np.array([0.0])
teach_idx = (np.searchsorted(s_teach, progress, side='left').clip(0, max(0, len(teach_points)-1))
             if len(teach_points) else np.array([], dtype=int))
repeat_idx = (np.searchsorted(s_repeat, progress, side='left').clip(0, max(0, len(repeat_points)-1))
              if len(repeat_points) else np.array([], dtype=int))

for i in range(max_len):
    if len(teach_points):
        ti = int(teach_idx[min(i, len(teach_idx)-1)])
        teach_pcd.points = o3d.utility.Vector3dVector(teach_points[ti].T)
        teach_pcd.paint_uniform_color((1.0, 0.0, 0.0))  # red
        if first:
            vis.add_geometry(teach_pcd)
        else:
            vis.update_geometry(teach_pcd)

    if len(repeat_points):
        ri = int(repeat_idx[min(i, len(repeat_idx)-1)])
        repeat_pcd.points = o3d.utility.Vector3dVector(repeat_points[ri].T)
        repeat_pcd.paint_uniform_color((0.0, 1.0, 0.0))  # green
        if first:
            vis.add_geometry(repeat_pcd)
        else:
            vis.update_geometry(repeat_pcd)

    print("Plotting submaps (red teach, green repeat)...  [space] to pause")
    t0 = time.time()
    while time.time() - t0 < 0.1 or paused:
        vis.poll_events()
        vis.update_renderer()

    first = False

vis.run()
vis.destroy_window()
