import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator, TemporalIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

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
    # g_utils.set_world_frame(test_graph, test_graph.get_vertex(root.id)) #, root.T_w_v)

    v_start = test_graph.root
    # print(f'root id: {root.id}')
    # v_start = test_graph.get_vertex(root.id)
    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))
    # path_matrix = vtr_path.path_to_matrix(test_graph, test_graph.root)#PriviledgedIterator(test_graph.get_vertex(root.id)))
    # print(path_matrix.shape)

    x = []
    y = []
    t = []

    # for v, e in TemporalIterator(v_start, to_goal=False): #####WAS PRIVILEDGED ITERATOR - CHANGING TO CHECK IF LOOP CLOSURE WORKED
    #     x.append(v.T_v_w.r_ba_ina()[0])
    #     y.append(v.T_v_w.r_ba_ina()[1])
    #     t.append(v.stamp / 1e9)
    path_len=0
    for v, e in PriviledgedIterator(v_start):
        # print(f'from {e.from_id}, to {e.to_id}, vertex {v.id}')
        # pdb.set_trace()
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        t.append(v.stamp / 1e9)
        path_len += np.linalg.norm(e.T.r_ba_ina())

        # if len(x) > 3050:
        #     break
    curvature = curvature_2d(np.hstack((x, y)))
    fname = f"virtr_urban" # path name
    repeat = {
        'x':x,
        'y':y,
        'c':curvature
    }

    # print(f"saving {fname}.pkl")
    # with open(f"curvature/{fname}.pkl", "wb") as f:
    #     pkl.dump(repeat, f)

    print(f"path length : {path_len}")
    plt.figure(0)
    plt.figure(0)
    plt.scatter(x, y, label="Teach", c=curvature, cmap='rainbow')#'#D86900')
    plt.colorbar()
    plt.axis('equal')
    # plt.savefig(f'plots/curvature/{fname}.png')
    plt.show()

