import cv2
import numpy as np
import cv_shared


def draw_pose(
    input_frame: cv2.Mat,
    is_success: bool,
    translation_vector: np.ndarray,
    rotation_vector: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    axis_length: float,
) -> None:
    text_color = (0, 0, 255)
    frame_height = input_frame.shape[0]
    frame_width = input_frame.shape[1]

    left_bottom_pos = (10, frame_height - 200)
    right_bottom_pos = (frame_width - 200, frame_height - 200)
    if not is_success:
        result_translation_vector: str = f"PnP is not solved: is_success: {is_success}"
        return cv_shared.put_on_frame(
            result_translation_vector,
            text_color,
            input_frame,
            left_bottom_pos,
        )
    # if is_success case
    x = translation_vector[0][0]
    y = translation_vector[1][0]
    z = translation_vector[2][0]
    result_translation_vector = f"""Vector from object's frame origin to the camera's frame origin
X_oc: {x:.5f}
Y_oc: {y:.5f}
Z_oc: {z:.5f}
|R_oc|: {np.sqrt(np.power(x,2)+np.power(y,2)+np.power(z,2)):.5f}"""
    display = cv_shared.put_on_frame(
        result_translation_vector,
        text_color,
        input_frame,
        left_bottom_pos,
    )
    result_rotation_vector = f"""Rotation vector from object's frame to the camera's frame
1: {rotation_vector[0][0]}
2: {rotation_vector[1][0]}
3: {rotation_vector[2][0]}"""
    display = cv_shared.put_on_frame(
        result_rotation_vector,
        text_color,
        display,
        right_bottom_pos,
    )
    return cv2.drawFrameAxes(
        display,
        camera_matrix,
        distortion_coeffs,
        rotation_vector,
        translation_vector,
        length=axis_length,
    )
