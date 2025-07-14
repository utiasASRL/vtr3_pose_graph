import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
import pylgmath.so3.operations as so3op

from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Plot Repeat Path',
                        description = 'Plots scatter of points to show path. Also calculates RMS error')
    parser.add_argument('graph', help="The filepath to the pose graph folder. (Usually /a/path/graph)")
    parser.add_argument('-f', '--filter', type=int, nargs="*", help="Select only some of the repeat runs. Default plots all runs.")
    parser.add_argument('-l', '--last', type=int, help="Plot only the last N repeats (overrides filter if set)") 
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
    t = []
    auto_path_len = 0

    # Number of initial vertices to exclude
    exclude_n_teach = int(os.getenv("EXCLUDE_VERTICES", 28)) #for whole big boi: 28, for without grassy: 28 - for solo parking: 5
    
    # Number of final vertices to exclude from the teach path
    exclude_n_teach_end = int(os.getenv("EXCLUDE_VERTICES_END", 53)) #for whole big boi: 53, , for without grassy: 483 - for solo parking: 10
    
    # Skip the first exclude_n_teach and last exclude_n_teach_end vertices in the teach path
    teach_iter = list(TemporalIterator(v_start))[exclude_n_teach:]
  
    if exclude_n_teach_end > 0:
        teach_iter = teach_iter[:-exclude_n_teach_end]

    for v, e in teach_iter:
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        t.append(v.stamp / 1e9)
    
    plt.figure(0)
    plt.scatter(x, y, label="Teach")
    plt.axis('equal')

    if args.filter is None:
        args.filter = [i+1 for i in range(test_graph.major_id)]
    
    if args.last is not None:
        args.filter = list(range(test_graph.major_id - args.last + 1, test_graph.major_id + 1))
    elif args.filter is None:
        args.filter = [i+1 for i in range(test_graph.major_id)]

    for i in range(test_graph.major_id):
        if i+1 not in args.filter:
            continue
        v_start = test_graph.get_vertex((i+1,0))

        pose_vec = np.zeros((6, 0))
        t = []
        dist = []
        p = []
        path_len = 0

        x = []
        y = []
        
        # Number of initial vertices to exclude
        exclude_n_repeat = int(os.getenv("EXCLUDE_VERTICES", 30)) #for whole big boi: 30, for without grassy: 30 - for solo parking: 0 
        
        # Number of final vertices to exclude from the repeat path
        exclude_n_repeat_end = int(os.getenv("EXCLUDE_VERTICES_END", 0)) #for whole big boi: 0, for without grassy: 430 - for solo parking: 60
       
        # Skip the first exclude_n vertices in the repeat path
        repeat_iter = list(TemporalIterator(v_start))[exclude_n_repeat:]
        if exclude_n_repeat_end > 0:
            repeat_iter = repeat_iter[:-exclude_n_repeat_end]
     
        for v, e in repeat_iter:
            pose_vec = np.hstack((pose_vec, np.vstack((v.T_v_w.r_ba_ina(), so3op.rot2vec(v.T_w_v.C_ba()) * 180/np.pi))))
            t.append(v.stamp / 1e9)
            dist.append(vtr_path.signed_distance_to_path(v.T_v_w.r_ba_ina(), path_matrix))
            path_len += np.linalg.norm(e.T.r_ba_ina())
            p.append(path_len)
        t = np.array(t)
        t_sort = np.argsort(t)
        t = t[t_sort]
        pose_vec = pose_vec[:, t_sort]
        print(t.shape)

        if t.shape[0] < 2 or v.taught:
            continue

        print(f"Path {i+1} was {path_len:.3f}m long with {len(t)} vertices")
        if len(t) < 2 or v.taught:
            continue

        plt.figure(0)
        plt.scatter(pose_vec[0], pose_vec[1], label=f"Repeat {i+1}")
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()

        plt.figure(5)
        # plt.plot(p, pose_vec[3], label=f"Roll Run {i+1}")
        # plt.plot(p, pose_vec[4], label=f"Pitch Run {i+1}")
        plt.plot(p, pose_vec[5], label=f"Yaw Run {i+1}")
        plt.title("Orientation")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle ($^\circ$)")
        plt.legend()

        plt.figure(1)
        plt.plot(p, pose_vec[0], label="X")
        plt.plot(p, pose_vec[1], label="Y")
        plt.plot(p, pose_vec[2], label="Z")
        plt.title("Position")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        
        #print(f"Path {i+1} was {path_len:.3f}m long with {len(x)} vertices")
        print(f"Path {i+1} took {t[-1] - t[0]:.1f}s to complete")

        auto_path_len += path_len

        plt.figure(2)
        rmse_int = np.sqrt(np.trapz(np.array(dist)**2, t) / (t[-1] - t[0]))
        rmse = np.sqrt(np.mean(np.array(dist)**2))
        print(rmse, rmse_int)

        plt.plot(p, dist, label=f"RMSE: {rmse:.3f}m for Repeat {i+1}")
        plt.legend()
        plt.ylabel("Path Tracking Error (m)")
        plt.xlabel("Time (s)")
        plt.title("Path Tracking Error")

        plt.figure(3)
        vx = np.gradient(pose_vec[0], t).squeeze()
        vy = np.gradient(pose_vec[1], t).squeeze()
        plt.plot(p, np.hypot(vx, vy), label=f"V for Repeat {i+1}")
        plt.legend()
        plt.ylabel("Velocity (m/s)")
        plt.xlabel("Time (s)")

        plt.figure(4)
        acc_x = np.gradient(vx, t).squeeze()
        acc_y = np.gradient(vy, t).squeeze()
        plt.plot(p, np.hypot(acc_x, acc_y), label=f"Acc for Repeat {i+1}")
        plt.legend()
        plt.ylabel("Acceleration (m/s^2)")
        plt.xlabel("Time (s)")
    print(f"Total repeat distance {auto_path_len:.2f} m")
    plt.show()