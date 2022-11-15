#!/usr/bin/python3

from py_image_proc_cuda import ImageProcCuda
import cv2
import pathlib
import rospkg


def main():
    # Load image
    rospack = rospkg.RosPack()
    image_path = rospack.get_path("ffcc_catkin") + "/data/alphasense.png"
    img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)

    # Set config files
    calib_file = rospack.get_path("image_proc_cuda") + "/config/alphasense_calib_example.yaml"
    color_calib_file = rospack.get_path("image_proc_cuda") + "/config/alphasense_color_calib_example.yaml"
    param_file = rospack.get_path("image_proc_cuda") + "/config/pipeline_params_example.yaml"

    # Create image Proc
    proc = ImageProcCuda(param_file, calib_file, color_calib_file)

    # Uncomment below to show calibration data
    # print("image_height:", proc.get_image_height())
    # print("image_width:", proc.get_image_width())
    # print("distortion_model:", proc.get_distortion_model())
    # print("camera_matrix:", proc.get_camera_matrix())
    # print("distortion_coefficients:", proc.get_distortion_coefficients())
    # print("rectification_matrix:", proc.get_rectification_matrix())
    # print("projection_matrix:", proc.get_projection_matrix())

    # Apply pipeline without modifying input
    img2 = proc.process(img, "bgr8")

    # Apply pipeline changing the input
    proc.apply(img, "bgr8")

    # show image
    cv2.imwrite("output_apply.png", img)
    cv2.imwrite("output_process.png", img2)


if __name__ == "__main__":
    main()
