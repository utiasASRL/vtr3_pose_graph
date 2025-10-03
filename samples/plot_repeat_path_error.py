import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='Plot Repeat Path',
                        description='Plots scatter of points to show path. Also calculates RMS error')
    parser.add_argument('graph', 
                        help="The filepath to the pose graph folder. (Usually /a/path/graph)")
    parser.add_argument('-r', '--run', type=int, required=True, 
                        help="Select a repeat run.")
    args = parser.parse_args()

    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    # Plot the teach path
    v_start = test_graph.root
    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))

    xt = []
    yt = []
    zt = []
    tt = []

    for v, e in PriviledgedIterator(v_start):
        xt.append(v.T_v_w.r_ba_ina()[0])
        yt.append(v.T_v_w.r_ba_ina()[1])
        zt.append(v.T_v_w.r_ba_ina()[2])
        tt.append(v.stamp / 1e9)

    plt.figure(0)
    plt.scatter(xt, yt, label="Teach", c='#D86900')
    plt.axis('equal')
    plt.grid()

    plt.figure(1)
    plt.scatter(xt, zt, label="Teach")

    # Process the repeat run: compute cumulative path length `p`
    vertex_start = test_graph.get_vertex((args.run, 0))

    repeat_x = []
    repeat_y = []
    repeat_z = []
    repeat_t = []
    p = []     # cumulative path length
    dist = []  # tracking error at each vertex
    path_len = 0

    for v, e in TemporalIterator(vertex_start):
        repeat_x.append(v.T_v_w.r_ba_ina()[0])
        repeat_y.append(v.T_v_w.r_ba_ina()[1])
        repeat_z.append(v.T_v_w.r_ba_ina()[2])
        repeat_t.append(v.stamp / 1e9)
        dist.append(vtr_path.signed_distance_to_path(v.T_v_w.r_ba_ina(), path_matrix))
        path_len += np.linalg.norm(e.T.r_ba_ina())
        p.append(path_len)

    max_error = max(abs(v) for v in dist)

    print(f"Path {args.run} was {path_len:.3f}m long")
    if len(repeat_t) > 2:
        c = [abs(v) for v in dist]

        # Overplot the repeat run on figure 0
        plt.figure(0)
        plt.scatter(repeat_x, repeat_y, label=f"Repeat {args.run} (Max Error: {max_error:.3f}m)", c=c)
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.colorbar(label="Lateral Error (m)")
        plt.legend()

        plt.figure(1)
        plt.title("Elevation of Path")
        plt.scatter(repeat_x, repeat_z, label=f"Repeat {args.run} (Max Error: {max_error:.3f}m)", c=c)
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.colorbar(label="Lateral Error (m)")
        plt.legend()

        plt.figure(2)
        plt.grid()
        rmse_int = np.sqrt(np.trapz(np.array(dist)**2, p) / (p[-1] - p[0]))
        rmse = np.sqrt(np.mean(np.array(dist)**2))
        plt.plot(p, dist, label=f"RMSE: {rmse:.3f}m, Max Error: {max_error:.3f}m for Repeat {args.run}")
        print(f"RMSE: {rmse:.3f}m for Repeat {args.run}")
        print(f"Max Error: {max(c):.3f}m for Repeat {args.run}")
        plt.legend()
        plt.ylabel("Path Tracking Error (m)")
        plt.xlabel("Path Length (m)")
        plt.title("Path Tracking Error")

        # Define the click event handler for figure 0 (Teach/Repeat scatter plot)
        def on_click_scatter(event):
            if event.inaxes:
                click_x, click_y = event.xdata, event.ydata
                distances = np.hypot(np.array(repeat_x) - click_x, np.array(repeat_y) - click_y)
                idx = np.argmin(distances)
                print(f"Clicked near point at path length: {float(p[idx]):.3f} m (Distance from click: {float(distances[idx]):.3f} m)")

        plt.figure(0)
        plt.gcf().canvas.mpl_connect('button_press_event', on_click_scatter)

        # New plot: Path Tracking Error over Time
        plt.figure(3)
        plt.grid()
        plt.plot(repeat_t, dist, label=f"Path Tracking Error over Time for Repeat {args.run}")
        plt.xlabel("Time (s)")
        plt.ylabel("Path Tracking Error (m)")
        plt.title("Path Tracking Error over Time")
        plt.legend()

        # Define click event for the new plot (figure 3)
        def on_click_error_over_time(event):
            if event.inaxes:
                click_time = event.xdata
                # Find the index closest in time to the click
                idx = np.argmin(np.abs(np.array(repeat_t) - click_time))
                print(f"Clicked at time: {repeat_t[idx]:.3f} s, cumulative path length: {p[idx]:.3f} m")

        plt.gcf().canvas.mpl_connect('button_press_event', on_click_error_over_time)

        plt.show()