import datetime
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Plot Repeat Path',
                        description = 'Plots scatter of points to show path. Also calculates RMS error')
    parser.add_argument('graph', help="The filepath to the pose graph folder. (Usually /a/path/graph)")
    args = parser.parse_args()

    factory = Rosbag2GraphFactory(args.graph)

    graph = factory.buildGraph()
    print(f"Graph {graph} has {graph.number_of_vertices} vertices and {graph.number_of_edges} edges")


    for i in range(graph.major_id + 1):
        v_start = graph.get_vertex((i,0))

        x = []
        y = []
        t = []
        dist = []
        path_len = 0

        for v, e in TemporalIterator(v_start):
            x.append(v.T_v_w.r_ba_ina()[0])
            y.append(v.T_v_w.r_ba_ina()[1])
            t.append(v.stamp / 1e9)
            path_len += np.linalg.norm(e.T.r_ba_ina())
        
        print(f"Path {i} was {path_len:.3f}m long with {len(x)} vertices and is a {'teach' if v.taught else 'repeat'}.")
        print(f"Path {i} started at {datetime.datetime.fromtimestamp(t[0])} took {t[-1] - t[0]:.1f}s to complete")


