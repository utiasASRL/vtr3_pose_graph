import open3d as o3d
import numpy as np

# Input and output file paths
input_ply = "point_clouds/pin_4_crop_sub.ply"
output_ply = "point_clouds/pin_4_filtered.ply"

# Read the point cloud
pcd = o3d.io.read_point_cloud(input_ply)

# Convert to numpy array
points = np.asarray(pcd.points)

pix4d_pc_base_height = -12
radar_height = 1.5
radar_angle = 2  # degrees
radar_range = 20  # meters

# Calculate centroid of x and y coordinates
centroid_x = np.mean(points[:, 0])
centroid_y = np.mean(points[:, 1])

# Radar position in the point cloud coordinate system
# Using centroid of points for x and y coordinates
radar_position = np.array([centroid_x, centroid_y, pix4d_pc_base_height + radar_height])

# Calculate horizontal distance of each point from radar in x-y plane
distances = np.sqrt((points[:, 0] - radar_position[0])**2 + 
                    (points[:, 1] - radar_position[1])**2)

# Calculate allowed z-range at each distance based on radar_angle
allowed_deviation = distances * np.tan(np.deg2rad(radar_angle))

# Calculate upper and lower bounds for each point
lower_bounds = radar_position[2] - allowed_deviation
upper_bounds = radar_position[2] + allowed_deviation

# Filter points that fall within the conic field of view
mask = (points[:, 2] >= lower_bounds) & (points[:, 2] <= upper_bounds) & (distances <= radar_range)
filtered_points = points[mask]

# Set z coordinates to 0
filtered_points[:, 2] = 0

# Create new point cloud
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# Optionally, copy colors if present
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

# Save the filtered point cloud
o3d.io.write_point_cloud(output_ply, filtered_pcd)