# VT&R3 Pose Graph Tools for Python

Read from the graph folder to load pose graphs including transformations and data associated with all the vertices.

The basic idea is to replicate the pose graph data structure used in [VT&R](https:/github.com/utiasASRL/vtr3).

Note that the current implementation does not support creating or writing to pose graphs.
This library is useful to load point clouds into PyTorch or open3d for machine learning or visualization offline. 

This library depends on ROS2 for reading the bag files used in VT&R. It is recomended that you run this repository inside of a [Docker container built for VT&R](https://github.com/utiasASRL/vtr3/wiki/Installation).

The samples contain tools for plotting paths. 

There is an optional open3d dependency for visualization. 