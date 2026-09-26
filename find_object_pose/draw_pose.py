import cv2
from find_object_pose.calculate_Euler_angels import EulerAngels
import numpy as np
import cv_shared


def draw_pose(
    input_frame: cv2.Mat,
    is_success: bool,
    translation_vector: np.ndarray,
    rotation_vector: np.ndarray,
    Euler_angels: EulerAngels,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    axis_length: float,
    distance_uom: str
) -> None:
    text_color = (0, 0, 255)
    additional_text_color = (49, 125, 215)

    frame_height = input_frame.shape[0]
    frame_width = input_frame.shape[1]

    center_bottom_position=(int(frame_width*0.25),frame_height)
    info: str = f"Camera calibration and calculations are done in {distance_uom}"
    cv_shared.put_on_frame(info, additional_text_color,input_frame, center_bottom_position)
    
    left_bottom_pos = (10, frame_height - 300)
    right_top_pos = (frame_width - 400, 40)
    if not is_success:
        result: str = f"PnP is not solved: is_success: {is_success}"
        return cv_shared.put_on_frame(
            result,
            text_color,
            input_frame,
            left_bottom_pos,
        )
    # if is_success case
    x = translation_vector[0][0]
    y = translation_vector[1][0]
    z = translation_vector[2][0]

    info_uom: str ='cm'
    # I assign 0 to factor to pay attention of the user 
    distance_2_info_uom_factor = 100.0 if distance_uom=='m' else 0.0

    result_translation_vector = f"""Vector between 
object's frame origin and
camera's frame origin ({info_uom})
X_oc: {x*distance_2_info_uom_factor:.2f}
Y_oc: {y*distance_2_info_uom_factor:.2f}
Z_oc: {z*distance_2_info_uom_factor:.2f}
|R_oc|: {np.sqrt(np.power(x,2)+np.power(y,2)+np.power(z,2))*distance_2_info_uom_factor:.2f}"""
    display = cv_shared.put_on_frame(
        result_translation_vector,
        text_color,
        input_frame,
        left_bottom_pos,
    )
    result_rotation_vector = f"""Rotational vector
1: {rotation_vector[0][0]:.2f}
2: {rotation_vector[1][0]:.2f}
3: {rotation_vector[2][0]:.2f}
    Euler angels 
between object's frame 
and the camera's frame 
(degrees)
!TO BE INVESTIGATED!
yaw (рыскание): {Euler_angels.yaw:.1f}
pitch (тангаж): {Euler_angels.pitch:.1f}
roll (крен)   : {Euler_angels.roll:.1f}"""
    display = cv_shared.put_on_frame(
        result_rotation_vector,
        text_color,
        display,
        right_top_pos,
    )
    return cv2.drawFrameAxes(
        display,
        camera_matrix,
        distortion_coeffs,
        rotation_vector,
        translation_vector,
        length=axis_length,
    )
