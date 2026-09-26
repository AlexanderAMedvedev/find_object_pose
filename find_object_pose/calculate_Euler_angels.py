import math
from typing import NamedTuple

import numpy as np
import cv2

class EulerAngels(NamedTuple):
    yaw: float
    pitch: float
    roll: float

def calculate_Euler_angels(rotational_vector_from_solvePnP: np.ndarray) -> EulerAngels:
    # Order of rotation (not sure)
    # 1) yaw [jo:]   - rotation around ?-axis (рыскание) 
    # 2) pitch       - rotation around ?-axis (тангаж)
    # 3) roll [reul] - rotation around ?-axis (крен)
    
    
    # Just copy from the previous project (to be checked)
    rotation_matrix, _ = cv2.Rodrigues(rotational_vector_from_solvePnP)
    sy = math.sqrt(
            rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
        )

    singular = sy < 1e-6

    if not singular:
            # original order
            # 1
            roll = math.atan2(
                rotation_matrix[2, 1], rotation_matrix[2, 2]
            )
            # 2
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            # 3
            yaw = math.atan2(
                rotation_matrix[1, 0], rotation_matrix[0, 0]
            )
    else:
            roll = math.atan2(
                -rotation_matrix[1, 2], rotation_matrix[1, 1]
            )
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            yaw = 0
    return EulerAngels(yaw=math.degrees(yaw), pitch=math.degrees(pitch), roll=math.degrees(roll) )
    