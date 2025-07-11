import os
import csv
import numpy as np
import argparse
from pylgmath import se3op 
import vtr_pose_graph.graph_utils as g_utils
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def export_absolute_transforms(graph_path, output_path):
    if os.path.isdir(output_path):
        output_path_mat = os.path.join(output_path, "absolute_transforms_mat_test.csv")
        output_path_abs = os.path.join(output_path, "absolute_transforms_test.csv")
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
    absolute_transforms_mat = []
    absolute_transforms = []

    for v, _ in PriviledgedIterator(v_start):
        # Extract absolute transform
        T_abs = v.T_v_w
        print(T_abs)
        timestamp = v.stamp / 1e9  # Convert nanoseconds to seconds

        # Extract absolute transform as 4x4 matrix
        T_abs_matrix = T_abs.matrix()
        absolute_transforms_mat.append([float(timestamp)] + T_abs_matrix.flatten().tolist())
        
        # Use pylgmath function tran2vec to convert the 4x4 transformation matrix to a vector
        abs_vector = np.array(se3op.tran2vec(T_abs_matrix).tolist())
        abs_vector = abs_vector.flatten()

        # Combine timestamp and the transformation vector
        absolute_transforms.append([float(timestamp)] + abs_vector.tolist())

    print("T_abs_mat", T_abs_matrix)
    print("abs transforms_mat", absolute_transforms_mat[1])
    print("abs_vec", abs_vector[1])
    print("abs vec transforms", absolute_transforms[1])


    # Save absolute transforms matrix to CSV file
    with open(output_path_mat, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        header = ['timestamp'] + [f'm{i}{j}' for i in range(4) for j in range(4)]
        csvwriter.writerow(header)
        # Write data
        csvwriter.writerows(absolute_transforms_mat)
        print(f"Saved absolute transforms matrix to {output_path_mat}")

    # # Save absolute transforms to CSV file
    # with open(output_path_abs, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     # Write header
    #     csvwriter.writerow(['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
    #     # Write data
    #     csvwriter.writerows(absolute_transforms)
    #     print(f"Saved absolute transforms to {output_path_abs}")

    # Plot the path using the absolute positions extracted from the pose graph.
    import matplotlib.pyplot as plt

    # Extract translation components from the flat 4x4 transformation matrices.
    # The translation is at indices 4, 8, and 12 corresponding to (0,3), (1,3), and (2,3)
    positions = np.array([[row[4], row[8], row[12]] for row in absolute_transforms_mat])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Absolute Pose Graph Path')
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

    export_absolute_transforms(args.graph, args.output)