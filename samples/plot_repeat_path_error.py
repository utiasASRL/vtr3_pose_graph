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
                        prog = 'Plot Repeat Path',
                        description = 'Plots scatter of points to show path. Also calculates RMS error')
    parser.add_argument('graph', help="The filepath to the pose graph folder. (Usually /a/path/graph)")
    parser.add_argument('-r', '--run', type=int, help="Select a repeat run.")
    args = parser.parse_args()

    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.root

    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))

    x = []
    y = []
    z = []
    t = []

    for v, e in PriviledgedIterator(v_start):
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        z.append(v.T_v_w.r_ba_ina()[2])
        t.append(v.stamp / 1e9)

    plt.figure(0)
    plt.scatter(x, y, label="Teach", c='#D86900')
    plt.axis('equal')
    plt.grid()

    plt.figure(1)
    plt.scatter(x, z, label="Teach")

    
    v_start = test_graph.get_vertex((args.run,0))

    x = []
    y = []
    z = []
    t = []
    dist = []
    path_len = 0

    for v, e in TemporalIterator(v_start):
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        z.append(v.T_v_w.r_ba_ina()[2])
        t.append(v.stamp / 1e9)
        dist.append(vtr_path.signed_distance_to_path(v.T_v_w.r_ba_ina(), path_matrix))
        path_len += np.linalg.norm(e.T.r_ba_ina())
    


    print(f"Path {args.run} was {path_len:.3f}m long")
    if len(t) > 2:
        c = [abs(v) for v in dist]

        plt.figure(0)
        plt.scatter(x, y, label=f"Repeat {args.run}", c=c)
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.colorbar(label="Lateral Error (m)")
        plt.legend()

        plt.figure(1)
        plt.title("Elevation of Path")
        plt.scatter(x, z, label=f"Repeat {args.run}", c=c)
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.colorbar(label="Lateral Error (m)")
        plt.legend()


        plt.figure(2)
        plt.grid()
        rmse = np.sqrt(np.trapz(np.array(dist[5:])**2, t[5:]) / (t[-1] - t[0]))

        plt.plot(t, dist, label=f"RMSE: {rmse:.3f}m for Repeat {args.run}")
        print(f"RMSE: {rmse:.3f}m for Repeat {args.run}")
        print(f"Max Error: {max(c):.3f}m for Repeat {args.run}")
        plt.legend()
        plt.ylabel("Path Tracking Error (m)")
        plt.xlabel("Time (s)")
        plt.title("Path Tracking Error")

        plt.show()

