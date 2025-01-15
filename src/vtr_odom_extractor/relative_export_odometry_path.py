import os
import csv
import numpy as np
from pylgmath import se3op # USE VEC2TRAN AND TRANVEC 
import argparse
import vtr_pose_graph.graph_utils as g_utils
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def export_relative_transforms(graph_path, output_path):
    # Check if output_path is a directory and append default filenames
    if os.path.isdir(output_path):
        output_path_mat = os.path.join(output_path, "relative_transforms_mat_test.csv")
        output_path_abs = os.path.join(output_path, "relative_transforms_test.csv")
    else:
        output_path_mat = output_path.replace(".csv", "_mat_test.csv")
        output_path_abs = output_path.replace(".csv", "_test.csv")

    # Load the pose graph
    print(f"Loading graph from {graph_path}...")
    factory = Rosbag2GraphFactory(graph_path)
    graph = factory.buildGraph()

    print(f"Graph {graph} has {graph.number_of_vertices} vertices and {graph.number_of_edges} edges")

    # Set world frame
    g_utils.set_world_frame(graph, graph.root)

    # Initialize iterator
    v_start = graph.root
    relative_transforms = []
    relative_transforms_mat = []


    for v, e in PriviledgedIterator(v_start):
        if e is not None:
            # Extract relative transform
            T_rel = e.T.inverse()
            print(T_rel)
            timestamp = v.stamp / 1e9  # Convert nanoseconds to seconds

            T_rel_matrix = T_rel.matrix()
            relative_transforms_mat.append([float(timestamp)] + T_rel_matrix.flatten().tolist())

            # Extract relative position and orientation
            rel_position = T_rel.r_ba_ina()
            rel_rotation = R.from_matrix(T_rel.C_ba()).as_euler('xyz', degrees=False)
            
            # # Adjust the y-coordinate to correct the flipping issue
            rel_position[1] = -rel_position[1]
            rel_rotation[1] = -rel_rotation[1]
            rel_rotation[2] = -rel_rotation[2]

            # Combine timestamp, relative position (x,y,z) and orientation (roll,pitch,yaw)
            relative_transforms.append([
                float(timestamp),
                float(rel_position[0]), float(rel_position[1]), float(rel_position[2]),
                float(rel_rotation[0]), float(rel_rotation[1]), float(rel_rotation[2])
            ])
    '''

    prev_vertex = None
    for v, e in PriviledgedIterator(v_start):
        if prev_vertex is not None:
            # Calculate relative transform between current vertex and previous vertex
            T_prev = prev_vertex.T_v_w
            T_curr = v.T_v_w
            T_rel = T_prev.inverse() * T_curr

            # Extract relative position and orientation
            rel_position = T_rel.r_ba_ina()
            rel_rotation = R.from_matrix(T_rel.C_ba()).as_euler('xyz', degrees=False)
            timestamp = v.stamp / 1e9  # Convert nanoseconds to seconds

            # Combine timestamp, relative position (x,y,z) and orientation (roll,pitch,yaw)
            relative_transforms.append([
                float(timestamp),
                float(rel_position[0]), float(rel_position[1]), float(rel_position[2]),
                float(rel_rotation[0]), float(rel_rotation[1]), float(rel_rotation[2])
            ])
        prev_vertex = v
    '''
    
    # Save relative transforms matrix to CSV file
    with open(output_path_mat, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        header = ['timestamp'] + [f'm{i}{j}' for i in range(4) for j in range(4)]
        csvwriter.writerow(header)
        # Write data
        csvwriter.writerows(relative_transforms_mat)
        print(f"Saved relative transforms matrix to {output_path_mat}")

    # Save relative transforms to CSV file
    with open(output_path_abs, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        # Write data
        csvwriter.writerows(relative_transforms)
        print(f"Saved relative transforms to {output_path_abs}")

    print(f"Relative transforms saved to {output_path}")
    
    # Compute the actual path from relative transformations
    actual_positions = []
    current_position = np.array([0.0, 0.0, 0.0])
    current_orientation = R.from_euler('xyz', [0.0, 0.0, 0.0])

    for transform in relative_transforms:
        timestamp, x, y, z, roll, pitch, yaw = transform
        rel_position = np.array([x, y, z])
        rel_orientation = R.from_euler('xyz', [roll, pitch, yaw])

        # Update current position and orientation
        current_position += current_orientation.apply(rel_position)
        current_orientation *= rel_orientation

        actual_positions.append(current_position.copy())

    actual_positions = np.array(actual_positions)

    # Plot the path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], label='Odometry Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Odometry Path')
    plt.show()

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

    export_relative_transforms(args.graph, args.output)


