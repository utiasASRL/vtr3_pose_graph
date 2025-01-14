import os
import csv
import numpy as np
import argparse
import vtr_pose_graph.graph_utils as g_utils
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def export_absolute_transforms(graph_path, output_path):
    # Check if output_path is a directory and append default filename
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "absolute_transforms.csv")

    # Load the pose graph
    print(f"Loading graph from {graph_path}...")
    factory = Rosbag2GraphFactory(graph_path)
    graph = factory.buildGraph()

    print(f"Graph {graph} has {graph.number_of_vertices} vertices and {graph.number_of_edges} edges")

    # Set world frame
    g_utils.set_world_frame(graph, graph.root)

    # Initialize iterator
    v_start = graph.root
    absolute_transforms = []
    current_pose = np.eye(4)  # Start at origin

    for v, e in PriviledgedIterator(v_start):
        if e is not None:
            # Extract absolute transform
            T_abs = v.T_v_w

            # Extract absolute position and orientation
            abs_position = T_abs.r_ba_ina()
            abs_rotation = R.from_matrix(T_abs.C_ba()).as_euler('xyz', degrees=False)
            timestamp = v.stamp / 1e9  # Convert nanoseconds to seconds

            # Combine timestamp, absolute position (x,y,z) and orientation (roll,pitch,yaw)
            absolute_transforms.append([
                float(timestamp),
                float(abs_position[0]), float(abs_position[1]), float(abs_position[2]),
                float(abs_rotation[0]), float(abs_rotation[1]), float(abs_rotation[2])
            ])
    print("first absolute transformation", absolute_transforms[0], absolute_transforms[1], absolute_transforms[2])

    # Save to CSV file
    with open(output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        # Write data
        csvwriter.writerows(absolute_transforms)
    
    # Save to CSV
    header = "timestamp,dx,dy,dz,droll,dpitch,dyaw"
    np.savetxt(output_path, absolute_transforms, delimiter=",", header=header, comments="")
    print(f"Relative odometry path saved to {output_path}")

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

    export_absolute_transforms(args.graph, args.output)