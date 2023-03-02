#!/usr/bin/python3

from py_image_proc_cuda import ImageProcCuda
import cv2
import pathlib
import rospkg


def main():
    # Load image
    rospack = rospkg.RosPack()
    image_path = rospack.get_path("image_proc_white_balance") + "/data/alphasense.png"
    img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)

    # Set config files
    calib_file = rospack.get_path("image_proc_cuda") + "/config/alphasense_calib_example.yaml"
    color_calib_file = rospack.get_path("image_proc_cuda") + "/config/alphasense_color_calib_example.yaml"
    param_file = rospack.get_path("image_proc_cuda") + "/config/pipeline_params_example.yaml"

    # Create image Proc
    proc = ImageProcCuda(param_file, calib_file, color_calib_file, False)

    # Uncomment below to show calibration data
    print("Original parameters:")
    print("  dist_image_height:", proc.get_dist_image_height())
    print("  dist_image_width:", proc.get_dist_image_width())
    print("  dist_distortion_model:", proc.get_dist_distortion_model())
    print("  dist_camera_matrix:", proc.get_dist_camera_matrix())
    print("  dist_distortion_coefficients:", proc.get_dist_distortion_coefficients())
    print("  dist_rectification_matrix:", proc.get_dist_rectification_matrix())
    print("  dist_projection_matrix:", proc.get_dist_projection_matrix())

    print("\nNew parameters:")
    print("  rect_image_height:", proc.get_rect_image_height())
    print("  rect_image_width:", proc.get_rect_image_width())
    print("  rect_distortion_model:", proc.get_rect_distortion_model())
    print("  rect_camera_matrix:", proc.get_rect_camera_matrix())
    print("  rect_distortion_coefficients:", proc.get_rect_distortion_coefficients())
    print("  rect_rectification_matrix:", proc.get_rect_rectification_matrix())
    print("  rect_projection_matrix:", proc.get_rect_projection_matrix())



    # Apply pipeline without modifying input
    img2 = proc.process(img, "bgr8")

    # Apply pipeline changing the input
    proc.apply(img, "bgr8")

    # show image
    cv2.imwrite("output_apply.png", img)
    cv2.imwrite("output_process.png", img2)


if __name__ == "__main__":
    main()
