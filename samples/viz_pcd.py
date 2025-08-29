import numpy as np
import open3d as o3d
import time
import struct
import csv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize point clouds and poses from .bin and .csv files")
    parser.add_argument('--bin', type=str, default="all_pointclouds.bin", help="Path to the .bin file containing point clouds")
    parser.add_argument('--csv', type=str, default="all_poses.csv", help="Path to the .csv file containing poses")
    parser.add_argument('--index', type=int, default=None, help="Index of the point cloud to visualize (default: all)")
    args = parser.parse_args()

    # Read pose and cloud metadata from CSV
    cloud_meta = []
    with open(args.csv, "r") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            cloud_meta.append(row)

    # Open the .bin file
    with open(args.bin, "rb") as fbin:
        # Setup Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="All Point Clouds")
        pcd = o3d.geometry.PointCloud()
        frame = None
        first = True
        for i, meta in enumerate(cloud_meta):
            if args.index is not None and i != args.index:
                continue
            offset = int(meta["offset_in_bin"])
            num_points = int(meta["num_points"])
            fbin.seek(offset)
            # Read number of points (uint32)
            n = struct.unpack('I', fbin.read(4))[0]
            assert n == num_points, f"Mismatch in number of points at index {i}"
            # Read all points (float32)
            pts = np.frombuffer(fbin.read(n * 12), dtype=np.float32).reshape((n, 3))
            # Read pose
            pose = np.array([float(meta[f"pose_{r}{c}"]) for r in range(4) for c in range(4)]).reshape((4, 4))
            # Invert the pose
            pose = np.linalg.inv(pose)

            # Set point cloud and frame
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color([0.2, 0.7, 0.9])
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=pose[:3, 3])

            if first:
                first = False
                vis.add_geometry(pcd)
                vis.add_geometry(frame)
            else:
                vis.update_geometry(pcd)
                vis.remove_geometry(frame, reset_bounding_box=False)
                vis.add_geometry(frame)
            t = time.time()
            # Display for 0.1 seconds
            while time.time() - t < 0.1:
                vis.poll_events()
                vis.update_renderer()
        vis.run()
        vis.destroy_window()
    print("Visualization complete.")
