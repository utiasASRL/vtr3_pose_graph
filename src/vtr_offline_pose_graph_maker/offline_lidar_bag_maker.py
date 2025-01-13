import rosbag2_py
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import struct

def numpy_to_pointcloud2(points: np.ndarray) -> PointCloud2:
    pc = PointCloud2()
    pc.header.frame_id = "map"
    pc.height = 1
    pc.width = len(points)
    pc.is_bigendian = False
    pc.is_dense = True
    pc.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    pc.point_step = 12
    pc.row_step = pc.point_step * points.shape[0]
    pc.data = struct.pack(f"{points.size}f", *points.flatten())
    return pc

writer = rosbag2_py.SequentialWriter()
storage_options = rosbag2_py.StorageOptions(uri='my_bag', storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(output_serialization_format='cdr')
writer.open(storage_options, converter_options)

# Define topics
writer.create_topic({
    'name': '/points',
    'type': 'sensor_msgs/PointCloud2',
    'serialization_format': 'cdr'
})

# Write data
for idx, scan in enumerate(my_scans):  # my_scans contains your filtered point clouds
    pc2 = numpy_to_pointcloud2(scan)
    writer.write('/points', pc2, timestamp=idx * 1e9)


#goal is to play the resultant bag while vtr does an offline teach pass for lidar