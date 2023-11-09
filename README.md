# VTR3 Pose Graph In Python

Read from the graph folder to load pose graphs including transformations and data associated with all the vertices.

The basic idea is to replicate the pose graph data structure used in (VT&R3)[https:/github.com/utiasASRL/vtr3].

Note that the current implementation does not support creating or writing to pose graphs.
This library is useful to load point clouds into PyTorch or open3d for machine learning or visualization offline. 

The samples contain tools for plotting paths and point clouds. 

There is an optional open3d dependency for visualization. 