#!/usr/bin/python3
# ArUco markers generated using https://chev.me/arucogen/
# 4x4, markers 0, 1, 2, 3
# marker size: 30 mm

import cv2
import cv2.aruco as aruco
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os.path
import os
from os.path import join
from scipy.optimize import least_squares, minimize, LinearConstraint, NonlinearConstraint
from tqdm import tqdm
from ruamel.yaml import YAML

COLOR_CHECKER_DIM = 24
SCALE_FACTOR = 4
TARGET_IMAGE_WIDTH = int(224 * SCALE_FACTOR)
TARGET_IMAGE_HEIGHT = int(160 * SCALE_FACTOR)
SQUARE_SIZE = int(30 * SCALE_FACTOR)
HALF_SQUARE_SIZE = int(SQUARE_SIZE / 2)
OFFSET = HALF_SQUARE_SIZE
MARGIN = int(2.5 * SCALE_FACTOR)
TARGET_IMAGE_SIZE = (
    (TARGET_IMAGE_WIDTH),
    (TARGET_IMAGE_HEIGHT),
)

# ArUco stuff
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters_create()
aruco_target_pts = np.array(
    [
        [0, 0],
        [TARGET_IMAGE_WIDTH, 0],
        [0, TARGET_IMAGE_HEIGHT],
        [TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT],
    ]
)


def show_image(image_bgr: np.array) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image_rgb)
    plt.show()


def show_calibration_result(ref_bgr: np.array, input_list: list, corr_list: np.array, save_path: str) -> None:
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    N = len(input_list)
    scale = 2
    fig, axs = plt.subplots(N, 3, figsize=(scale * 3, scale * N))
    for i in range(N):
        input_rgb = cv2.cvtColor(input_list[i], cv2.COLOR_BGR2RGB)
        corr_rgb = cv2.cvtColor(corr_list[i], cv2.COLOR_BGR2RGB)
        axs[i][0].imshow(input_rgb)
        axs[i][1].imshow(corr_rgb)
        axs[i][2].imshow(ref_rgb)

        for ax in axs[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(join(save_path, "calibrated_images.png"))


def save_calibration_matrix(filename: str, C: np.array) -> None:
    pass


def apply_color_correction(color_correction: np.matrix, img: np.array) -> np.array:
    C = color_correction["matrix"]
    b = color_correction["bias"]

    # change from HWC to CHW (channels first)
    img_chw = np.einsum("ijk->kij", img)
    # reshape as a vector of channels
    img_channels = img_chw.reshape(3, -1)
    # apply transformation
    img_channels = C @ img_channels + 255 * b
    # reshape
    img_chw = img_channels.reshape(img.shape)
    # Return to opencv's HWC
    return np.einsum("kij->ijk", img_chw)


def get_color_centroids(img: np.array, dim=COLOR_CHECKER_DIM) -> np.array:
    # Compute ARUCO coordinates
    aruco_corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    frame_markers = aruco.drawDetectedMarkers(img.copy(), aruco_corners, ids)

    if len(aruco_corners) != 4:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp_img = cv2.filter2D(img, -1, kernel)
        aruco_corners, ids, _ = aruco.detectMarkers(sharp_img, aruco_dict, parameters=aruco_params)
        frame_markers = aruco.drawDetectedMarkers(sharp_img.copy(), aruco_corners, ids)

        if len(aruco_corners) != 4:
            # print(f"Couldn't detect all the markers")
            return [], [], False

    # Create list of aruco center points
    try:
        aruco_center_pts = []
        for i in range(4):
            # find marker
            idx = list(ids.squeeze()).index(i)
            aruco_center_pts.append(aruco_corners[idx][-1].mean(axis=0))
        aruco_center_pts = np.array(aruco_center_pts)
    except:
        return [], [], False

    # Compute homography
    aruco_corner_pts = np.concatenate([x[0] for x in aruco_corners])
    H, mask = cv2.findHomography(aruco_center_pts, aruco_target_pts, cv2.RANSAC, 5.0)
    img_cropped = cv2.warpPerspective(img, H, TARGET_IMAGE_SIZE)
    img_cropped = img_cropped[OFFSET : TARGET_IMAGE_HEIGHT - OFFSET, OFFSET : TARGET_IMAGE_WIDTH - OFFSET]
    # show_image(img_cropped)

    # Compute the median for all pixels within each blob
    rgb_centroids = []
    x = int(MARGIN + HALF_SQUARE_SIZE)
    y = int(MARGIN + HALF_SQUARE_SIZE)
    s = int(HALF_SQUARE_SIZE * 0.5)
    d = int(SQUARE_SIZE + MARGIN)
    for i in range(4):  # marker has 24 squares
        x = int(MARGIN + HALF_SQUARE_SIZE)
        for j in range(6):
            # print(f"i: {i}, j: {j}, x={x}, y={y}")
            # centroid = img_cropped[y-s:y+s, x-s:x+s].copy().reshape(-1, 3) #.mean(axis=(0,1))
            centroid = np.median(img_cropped[y - s : y + s, x - s : x + s], axis=(0, 1))
            rgb_centroids.append(centroid)
            # img_cropped[y-s:y+s, x-s:x+s] = img_cropped[y-s:y+s, x-s:x+s] * 0
            x = x + d
        y = y + d
    # show_image(img_cropped)

    # Return the centroids
    return np.array(rgb_centroids), img_cropped, True
    # return np.concatenate(rgb_centroids)


def find_color_calibration(input: np.array, reference: np.array, loss="linear", compute_bias=False) -> np.array:
    # check input size
    assert input.shape == reference.shape

    # transpose input
    input = input.transpose()
    reference = reference.transpose()

    # regularization factor
    reg = 10

    # build optimization costs
    def fun(x: np.array):
        # sum || input * C - reference ||^2
        C = np.copy(x[:9].reshape((3, 3)))
        bias = np.copy(x[9:].reshape((3, 1)))
        color_corrected = np.matmul(C, input)

        if compute_bias:
            color_corrected = color_corrected + 255 * bias
        
        # Return residual
        return np.linalg.norm(color_corrected - reference)  # + reg * np.linalg.norm(x)

    def constraint(x: np.array):
        x = x.reshape((3, 3))
        return np.matmul(x, input).flatten()

    nonlinear_constraint = NonlinearConstraint(constraint, 0, 255)

    # minimize
    x0 = np.zeros((12,))
    x0[:9] = np.eye(3).flatten() * 0.1
    sol = least_squares(fun, x0, loss=loss)
    # sol = minimize(fun, x0, constraints=[nonlinear_constraint])

    # Recover calibration matrix and bias
    C = sol.x[:9].reshape((3, 3)).astype(np.float32)
    b = sol.x[9:].reshape((3, 1)).astype(np.float32)

    # Return color calibration matrix C
    return {"matrix": C, "bias": b, "sol": sol}


def main(*arg, **args):
    parser = argparse.ArgumentParser(description="Performs color calibration between 2 images, using ArUco 4X4")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input image (to be calibrated), or folder with reference images"
    )
    parser.add_argument("-r", "--ref", type=str, required=True, help="Reference image to perform the calibration")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Output path to store the file. Default: current path",
        default=os.path.abspath(os.getcwd()),
    )
    parser.add_argument("-p", "--prefix", type=str, help="Prefix for the calibration file. Default: none", default="")
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss used in the optimization. Options: 'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'",
        default="linear",
    )
    parser.add_argument(
        "--compute_bias",
        help="If bias should be computed",
        action="store_true",
    )

    args = parser.parse_args()

    # Open input images
    in_images = []
    if os.path.exists(args.input):
        if os.path.isfile(args.input):
            in_images.append(cv2.imread(args.input))
        elif os.path.isdir(args.input):
            # Check all files in dir
            images = [str(x) for x in Path(args.input).glob("*")]
            print("Loading images...")
            for im in tqdm(images):
                try:
                    img = cv2.imread(im)
                    if img is not None:
                        in_images.append(img)
                    else:
                        continue
                except Exception as e:
                    continue
    else:
        raise f"{args.input} does not exist"

    # Open reference image
    if os.path.exists(args.ref):
        ref_img = cv2.imread(args.ref)
    else:
        raise f"{args.ref} does not exist"

    # Extract color marker centroids
    # Input images
    print("Extracting color markers from input images...")
    centroids_in_list = []
    cropped_in_list = []
    for im in tqdm(in_images):
        centroid, crop_in, valid = get_color_centroids(im, dim=COLOR_CHECKER_DIM)
        if valid:
            centroids_in_list.append(centroid)
            cropped_in_list.append(crop_in)

    # Reference
    print("Extracting color markers from reference image...")
    centroid_ref, cropped_ref, valid = get_color_centroids(ref_img)
    # Check if the reference samples are valid
    if not valid:
        raise f"Failed to extract markers from the reference. Please check the reference image"

    # Fill reference data
    centroids_ref_list = [centroid_ref] * len(centroids_in_list)

    # Create training data
    centroids_in = np.concatenate(centroids_in_list, axis=0)
    centroids_ref = np.concatenate(centroids_ref_list, axis=0)

    # Find color calibration matrix
    print("Optimizing color correction matrix...", end="")
    sol = find_color_calibration(centroids_in, centroids_ref, loss=args.loss, compute_bias=args.compute_bias)
    print(f"done. Cost: {sol['sol']['cost']}")

    # Apply calibration
    cropped_corr_list = [apply_color_correction(sol, img).astype(np.uint8) for img in cropped_in_list]

    # Show result
    print("Generating visualization of calibrated images...")
    show_calibration_result(cropped_ref, cropped_in_list, cropped_corr_list, save_path=args.output_path)

    # Save as YAML file
    calib_str = f"""color_calibration_matrix:\n  rows: 3\n  cols: 3\n  data: {str([x for x in sol["matrix"].flatten()])}"""
    calib_str += f"""\ncolor_calibration_bias:\n  rows: 3\n  cols: 1\n  data: {str([x for x in sol["bias"].flatten()])}"""
    # Print calibration putput
    print(calib_str)
    
    yaml = YAML()
    yaml.width = 200
    output_file = os.path.join(args.output_path, "color_calibration.yaml")
    print(f"Saving calibration to {output_file}")
    with open(output_file, "w") as out_file:
        yaml.dump(yaml.load(calib_str), out_file)


if __name__ == "__main__":
    main()
