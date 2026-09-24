from pathlib import Path
import cv2
import get_a_priori_objects_points
import cv_shared
import find_n_points
import numpy as np
import get_camera_intrinsics

from find_object_pose import draw_pose

DEBUG = True
DEBUG_FILEPATH = "find_object_pose_debug.output"
#
A_PRIORI_POINTS_FILEPATH = (
    Path(__file__).resolve().parent.parent
    / "data_find_back_holes/a_priori_objects_points.json"
)
#
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "camera_config/camera_config.json"
)
DEFAULT_VIDEO_SOURCE = 0
MAX_READ_FAILURES = 30
#
CAMERA_INTRINSICS_FILEPATH = (
    Path(__file__).resolve().parent.parent
    / "data_find_back_holes/camera_calibration.json"
)
USE_CANNY_EDGE_DETECTOR = False
REDUCE_NOISE = True
BINARIZE_THRESHOLD = 48
MIN_AREA_PIXELS = 750
MIN_SOLIDITY = 0.95
MAX_SOLIDITY = 0.995
#
AXIS_LENGTH=0.06 # depends on the dimension of the apriori points


def main() -> None:
    prefix = "main (find_object_pose dependency example app)"
    if DEBUG:
        open(DEBUG_FILEPATH, "w").close()
    # 1 find the coordinates of the chosen N points in the chosen object's coordinate system
    object_points_in_objects_coordinate_system = get_a_priori_objects_points.get_a_priori_objects_points_pipeline(
        A_PRIORI_POINTS_FILEPATH,
        DEBUG,
        debug_filepath=DEBUG_FILEPATH,
    )  # dtype=np.float32 (set within the dependency for conveniency within the dependency)
    # 2 get the source of frames
    cap = cv_shared.open_video_capture(
        cv_shared.load_video_source(CONFIG_PATH, DEFAULT_VIDEO_SOURCE)
    )
    # 3 get the intrinsic properties of the camera
    camera_matrix_elements, dist_coeffs = (
        get_camera_intrinsics.get_camera_intrinsics_pipeline(
            CAMERA_INTRINSICS_FILEPATH,
            DEBUG,
            debug_filepath=DEBUG_FILEPATH,
        )
    )
    # construct the camera matrix as defined in
    # https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html
    # ( fx 0 cx )
    # ( 0 fy cy )
    # ( 0 0  1  )
    fx, fy = camera_matrix_elements[0], camera_matrix_elements[1]
    cx, cy = camera_matrix_elements[2], camera_matrix_elements[3]
    camera_matrix = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        dtype=np.float32,
    )
    dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
    # start to analyse the frames
    read_failures = 0
    while True:
        isOk, frame = cap.read()

        if not isOk:
            read_failures += 1
            if read_failures >= MAX_READ_FAILURES:
                print("The stream does not provide frames")
                break
            # Пауза перед следующей попыткой; waitKey заодно прокачивает очередь
            # событий окна, поэтому оно не «зависает», и даёт выйти по 'q'
            # if cv2.waitKey(READ_RETRY_DELAY_MS) & 0xFF == ord('q'):
            #    break
            continue
        else:
            read_failures = 0

        cv_shared.append_value_to_file(f"{prefix} ---next frame---", DEBUG_FILEPATH)
        # 4 find the coordinates of N points on the matrix frame
        camera_matrix_coordinates_of_virtual_angles, _, __ = (
            find_n_points.find_n_points_pipeline(
                frame=frame,
                debug=DEBUG,
                debug_filepath=DEBUG_FILEPATH,
                use_canny_edge_detector=USE_CANNY_EDGE_DETECTOR,
                reduce_noise=REDUCE_NOISE,
                binarize_threshold=BINARIZE_THRESHOLD,
                min_area_pixels=MIN_AREA_PIXELS,
                min_solidity=MIN_SOLIDITY,
                max_solidity=MAX_SOLIDITY,
            )
        )
        display = find_n_points.draw_result(
            frame,
            None,
            None,
            camera_matrix_coordinates_of_virtual_angles,
        )
        # 5 find the translation vector (3) and rotation angles (3) - 6 parameters total - from the object's
        #   coordinate system to the camera's coordinate system
        if camera_matrix_coordinates_of_virtual_angles is not None:
            camera_matrix_coordinates_of_virtual_angles = np.array(
                camera_matrix_coordinates_of_virtual_angles, dtype=np.float32
            )
            is_success, rotation_vector, translation_vector = cv2.solvePnP(
                object_points_in_objects_coordinate_system,
                camera_matrix_coordinates_of_virtual_angles,
                camera_matrix,
                dist_coeffs,
            )
            # 6 draw the object's coordinate system
            # and put as text the t_vec and r_vec
            display = draw_pose(
                display,
                is_success,
                translation_vector,
                rotation_vector,
                camera_matrix,
                dist_coeffs,
                axis_length=AXIS_LENGTH
            )
            # 7 write the obtained 6 parameters to move from the object's coordinate system to camera's coordinate system
            if DEBUG:
                cv_shared.append_value_to_file(
                    f"{prefix} \n\
                    solvePnP status: {is_success}\n\
                    rotation_vector:\n {rotation_vector}\n\
                    translation_vector:\n {translation_vector}",
                    DEBUG_FILEPATH,
                )

        cv2.imshow(f"{prefix}", display)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
