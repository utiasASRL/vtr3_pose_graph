import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import open3d as o3d
import numpy as np
import argparse
import time
import tempfile
import subprocess
import shutil
from pathlib import Path
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
from vtr_utils.plot_utils import extract_map_from_vertex

def get_vertex_world_pose(test_graph, v):
    """Returns (position [3,], T_world_v [4x4]) of vertex in world frame."""
    T_world_v = v.T_v_w.inverse().matrix()  # 4x4
    position = T_world_v[:3, 3]
    return position, T_world_v

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Plot Teach and Repeat Submaps',
        description='Plots submaps from teach and repeat overlaid. Also calculates RMS error')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="Path to the pose graph folder")
    parser.add_argument('-r', '--run', type=int, help="Select a repeat run.")
    parser.add_argument('--record', metavar='OUTPUT.mp4', default=None,
                        help="Record the visualization to an MP4 file (requires ffmpeg)")
    parser.add_argument('--no-vis', action='store_true',
                        help="Skip the interactive visualization window (useful with --record headlessly)")
    args = parser.parse_args()

    # --- Recording setup ---
    recording = args.record is not None
    frame_dir = None
    frame_index = [0]

    if recording:
        if not shutil.which('ffmpeg'):
            print("ERROR: ffmpeg not found in PATH. Install it with: sudo apt install ffmpeg")
            exit(1)
        frame_dir = tempfile.mkdtemp(prefix='vtr_record_')
        print(f"Recording frames to {frame_dir}, will encode to {args.record}")

    def capture_frame(vis):
        if recording:
            path = os.path.join(frame_dir, f"frame_{frame_index[0]:06d}.png")
            vis.capture_screen_image(path, do_render=True)
            frame_index[0] += 1

    def encode_video():
        out = args.record
        # Crop height to even number if needed, use ultrafast for speed
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', os.path.join(frame_dir, 'frame_%06d.png'),
            '-vf', 'scale=iw:trunc(ih/2)*2',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-movflags', '+faststart',
            out
        ]
        print(f"Encoding {frame_index[0]} frames -> {out}")
        subprocess.run(cmd, check=True)
        shutil.rmtree(frame_dir)
        print(f"Saved: {out}")

    # --- Build graph ---
    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)
    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")
    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start_repeat = test_graph.get_vertex((args.run, 0))

    paused = False
    def toggle(vis):
        global paused
        paused = not paused
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(' '), toggle)
    vis.create_window(visible=not args.no_vis)
    vis.poll_events()
    vis.update_renderer()
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = ((1.0, 10, 1.0))

    # --- Gather and plot ALL teach submaps across ALL teach runs upfront ---
    all_teach_points = []
    all_teach_poses = []
    for run_id in range(test_graph.major_id + 1):
        v_start_teach = test_graph.get_vertex((run_id, 0))
        vertices = list(TemporalIterator(v_start_teach))
        vertices_to_plot = vertices[:-10] if len(vertices) > 10 else vertices
        for v, e in vertices_to_plot:
            new_points, _ = extract_map_from_vertex(test_graph, v)
            if new_points is not None and new_points.size > 0:
                cxt_points = (new_points[2,:] > np.percentile(new_points[2,:], 90))
                position, T_world_v = get_vertex_world_pose(test_graph, v)
                all_teach_points.append(new_points[:, ~(cxt_points)])
                all_teach_poses.append((position, T_world_v))

    if not all_teach_points:
        print("No teach points found, exiting.")
        vis.destroy_window()
        exit()

    base_color = np.array([131/255.0, 152/255.0, 189/255.0])
    dark = base_color * 0.2
    bright = np.clip(base_color * 1.5, 0, 1)
    elev_cmap = LinearSegmentedColormap.from_list('elev', [dark, base_color, bright])

    merged_teach = np.hstack(all_teach_points)
    teach_pcd = o3d.geometry.PointCloud()
    teach_pcd.points = o3d.utility.Vector3dVector(merged_teach.T)

    z = merged_teach[2, :]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = plt.cm.Blues(z_norm)[:, :3]
    teach_pcd.colors = o3d.utility.Vector3dVector(colors)

    for position, T_world_v in all_teach_poses:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        frame.paint_uniform_color([0.0, 0.0, 0.0])
        frame.transform(T_world_v)
        vis.add_geometry(frame)

    vis.add_geometry(teach_pcd)
    vis.poll_events()
    vis.update_renderer()
    capture_frame(vis)  # capture static teach frame
    print(f"Plotted full teach map: {merged_teach.shape[1]} points from {len(all_teach_points)} submaps.")

    # --- Gather repeat submaps ---
    repeat_points = []
    all_repeat_poses = []
    vertices = list(TemporalIterator(v_start_repeat))
    vertices_to_plot = vertices[:-10] if len(vertices) > 10 else vertices
    for v, e in vertices_to_plot:
        new_points, map_ptr = extract_map_from_vertex(test_graph, v)
        new_points[2,:] += 0.2
        position, T_world_v = get_vertex_world_pose(test_graph, v)
        if new_points is not None and new_points.size > 0:
            repeat_points.append(new_points)
            all_repeat_poses.append((position, T_world_v))

    print(f"Streaming {len(repeat_points)} repeat submaps (green), one at a time over teach map...")

    # --- Stream repeat submaps ---
    repeat_pcd = o3d.geometry.PointCloud()
    repeat_added = False

    for i, (pts, (position, T_world_v)) in enumerate(zip(repeat_points, all_repeat_poses)):
        if i % 2 == 0:
            continue
        repeat_pcd.points = o3d.utility.Vector3dVector(pts.T)
        z_values = pts[2, :]
        z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min()) / 2 + 0.5

        cmap = plt.get_cmap('Reds')
        colors = cmap(z_norm)[:, :3]
        repeat_pcd.colors = o3d.utility.Vector3dVector(colors)

        if not repeat_added:
            vis.add_geometry(repeat_pcd, reset_bounding_box=True)
            repeat_added = True
        else:
            vis.update_geometry(repeat_pcd)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            axis.transform(T_world_v)

            position = np.array(position).flatten()
            forward  = T_world_v[:3, 0].flatten()
            up       = T_world_v[:3, 2].flatten()
            eye = position - forward * 10.0 + up * 10.0
            front = eye - position
            front = front / np.linalg.norm(front)

            ctr = vis.get_view_control()
            ctr.set_lookat(position.tolist())
            ctr.set_front(front.tolist())
            ctr.set_up(up.tolist())
            ctr.set_zoom(0.1)
            vis.add_geometry(axis, reset_bounding_box=False)

        # Render loop — capture frames at ~30fps when recording
        t0 = time.time()
        frame_interval = 1.0 / 30.0
        last_capture = 0.0
        while time.time() - t0 < 0.1 or paused:
            vis.poll_events()
            vis.update_renderer()
            now = time.time() - t0
            if recording and (now - last_capture) >= frame_interval:
                capture_frame(vis)
                last_capture = now

    if recording:
        # Capture a few final frames so the last pose holds on screen
        for _ in range(30):
            vis.poll_events()
            vis.update_renderer()
            capture_frame(vis)

    print("Done. Close window to exit.")

    if not args.no_vis:
        vis.run()

    vis.destroy_window()

    if recording:
        encode_video()