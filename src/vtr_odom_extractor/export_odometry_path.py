import os
import numpy as np
import argparse
import vtr_pose_graph.graph_utils as g_utils
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
import vtr_regression_testing.path_comparison as vtr_path


def export_odometry_path(graph_path, output_path):
    # Check if output_path is a directory and append default filename
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "odometry_path.csv")

    # Load the pose graph
    print(f"Loading graph from {graph_path}...")
    factory = Rosbag2GraphFactory(graph_path)
    graph = factory.buildGraph()

    print(f"Graph {graph} has {graph.number_of_vertices} vertices and {graph.number_of_edges} edges")

    # Set world frame
    g_utils.set_world_frame(graph, graph.root)

    # Initialize iterator
    v_start = graph.root
    path_matrix = vtr_path.path_to_matrix(graph, PriviledgedIterator(v_start))
    print(f"Path matrix shape: {path_matrix.shape}")

    # Extract odometry path
    odometry_data = []
    for v, e in PriviledgedIterator(v_start):
        # Extract position (r_ba_ina) and timestamp
        position = v.T_v_w.r_ba_ina()  # Ensure position is always 3 elements (x, y, z)
        timestamp = v.stamp / 1e9  # Convert nanoseconds to seconds
        odometry_data.append([timestamp, position[0], position[1], position[2]])

    # Ensure all rows in odometry_data have the same length
    odometry_data = np.array(odometry_data, dtype=float)


    # Save to CSV
    header = "timestamp,x,y,z"
    np.savetxt(output_path, odometry_data, delimiter=",", header=header, comments="")
    print(f"Odometry path saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export odometry path from pose graph.")
    parser.add_argument("-g", "--graph", required=True, help="Path to the pose graph directory.")
    parser.add_argument("-o", "--output", required=True, help="File path or directory to save the exported odometry file.")
    args = parser.parse_args()

    # Ensure the output directory exists if a directory is provided
    if os.path.isdir(args.output):
        print(f"Output directory exists: {args.output}")
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    export_odometry_path(args.graph, args.output)
