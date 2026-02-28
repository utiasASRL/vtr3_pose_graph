import numpy as np
from pylgmath.se3.transformation_with_covariance import TransformationWithCovariance

from vtr_pose_graph import INVALID_ID

EDGE_TYPE_TEMPORAL = 0
EDGE_TYPE_SPATIAL = 1
EDGE_TYPE_UNKNOWN = 2

EDGE_MODE_AUTONOMOUS = 0
EDGE_MODE_MANUAL = 1
EDGE_MODE_UNKNOWN = 2

class Edge:

    __id_source = 1


    def __init__(self, edge_msg=None):
        self.id = Edge.__id_source
        if edge_msg:
            self.type = edge_msg.type.type
            self.mode = edge_msg.mode.mode
            self.from_id = edge_msg.from_id
            self.to_id = edge_msg.to_id
            self.T = TransformationWithCovariance(xi_ab=np.array(edge_msg.t_to_from.xi).reshape(6, 1))
            if edge_msg.t_to_from.cov_set:
                self.T.set_covariance(np.array(edge_msg.t_to_from.cov).reshape((6, 6)))
        else:
            self.type = EDGE_TYPE_UNKNOWN
            self.mode = EDGE_MODE_UNKNOWN
            self.from_id = INVALID_ID
            self.to_id = INVALID_ID
            self.T = TransformationWithCovariance()
        Edge.__id_source += 1

    def is_teach(self):
        return self.mode == EDGE_MODE_MANUAL

    def is_repeat(self):
        return self.mode == EDGE_MODE_AUTONOMOUS

    def is_spatial(self):
        return self.type == EDGE_TYPE_SPATIAL

    def is_temporal(self):
        return self.type == EDGE_TYPE_TEMPORAL

    def __repr__(self):
        return f"{'Taught' if self.is_teach() else 'Repeat'} {'Temporal' if self.is_temporal() else 'Spatial'} Edge {self.from_id} -> {self.to_id}"