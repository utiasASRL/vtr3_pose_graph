from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_utils import collapse_pose_graph

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

    topology_only = collapse_pose_graph(graph.get_privileged_subgraph())
    print(f"Topology {topology_only} has {topology_only.number_of_vertices} vertices and {topology_only.number_of_edges} edges")
    for e in topology_only._edges.values():
        print(f"{e} with xi {e.T}")
