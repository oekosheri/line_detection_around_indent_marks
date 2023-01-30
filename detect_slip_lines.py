import os, sys
import numpy as np
import cv2
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def preprocess(image_address):
    raw_im = tf.io.read_file(image_address)
    image = tf.image.decode_png(raw_im, channels=1)
    input_image = tf.cast(image, tf.float32) / 255.0
    input_image = tf.image.resize_with_pad(
        input_image, 768, 1024, antialias=False
    )
    input_image = np.expand_dims(input_image, 0)
    return input_image


import cv2


def find_corners(prediction):
    gray = cv2.GaussianBlur(prediction, (7, 7), 0)
    # Find the contours
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # detect the shapes.
    # corners = {'corner':[], 'area':[]}
    corner = []
    area = 0
    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            if cv2.contourArea(cnt) > area:
                area = cv2.contourArea(cnt)
                # print(area)
                corner = np.reshape(approx, (3, 2))

    return corner


def edge_quad_points(corners, H=100):

    # sort on basis of x component
    corns = corners[corners[:, 0].argsort()]
    # corners
    A = corns[0]
    B = corns[1]
    C = corns[2]
    # edges
    ab = np.c_[np.reshape(B - A, (1, 2)), np.zeros((1, 1))]
    ac = np.c_[np.reshape(C - A, (1, 2)), np.zeros((1, 1))]
    bc = np.c_[np.reshape(C - B, (1, 2)), np.zeros((1, 1))]

    # normals
    if B[1] < A[1]:
        z_1 = np.array([0, 0, 1])
    else:
        z_1 = np.array([0, 0, -1])

    n_ab = np.squeeze(np.cross(ab, z_1))
    n_ab = n_ab / np.linalg.norm(n_ab)

    n_ac = np.squeeze(np.cross(ac, -z_1))
    n_ac = n_ac / np.linalg.norm(n_ac)

    n_bc = np.squeeze(np.cross(bc, z_1))
    n_bc = n_bc / np.linalg.norm(n_bc)

    # 4 points of rectangles
    points_1 = (
        np.ceil(B + n_ab[:2] * H).astype(int),
        B,
        A,
        np.ceil(A + n_ab[:2] * H).astype(int),
    )
    points_2 = (
        A,
        C,
        np.ceil(C + n_ac[:2] * H).astype(int),
        np.ceil(A + n_ac[:2] * H).astype(int),
    )
    points_3 = (
        B,
        C,
        np.ceil(C + n_bc[:2] * H).astype(int),
        np.ceil(B + n_bc[:2] * H).astype(int),
    )
    return points_1, points_2, points_3


def region_of_interest(p1, p2, p3, input_img):

    blank = np.zeros(input_img.shape[:2], dtype="uint8")
    mask = cv2.fillPoly(blank, np.array([p1]), 255, 1)
    mask = cv2.fillPoly(blank, np.array([p2]), 255, 1)
    mask = cv2.fillPoly(blank, np.array([p3]), 255, 1)

    # Passing the mask to the bitwise_and gives intersection point of the mask and the image
    maskimage = cv2.bitwise_and(input_img, input_img, mask=mask)
    return maskimage


def region_sl_coords(p1, p2, p3):
    coords = []
    slopes = []
    for p in [p1, p2, p3]:
        p = np.array(p)
        x1, y1 = p[0]
        x2, y2 = p[1]
        x3, y3 = p[2]
        m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
        m2 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else np.inf
        slopes.append([m1, m2])
        coords.append(p)

    return np.concatenate(slopes), np.concatenate(coords)


def detect_lines(
    img,
    p1,
    p2,
    p3,
    gauss_k=5,
    low_th=1,
    high_th=100,
    min_vote=50,
    min_line_length=30,
    max_line_gap=15,
    atol_p=30,
    atol_m=5,
):
    sl, points = region_sl_coords(p1, p2, p3)
    rho = 1
    theta = np.pi / 180
    img = np.uint8(img * 255.0)
    blurred = cv2.GaussianBlur(img, (gauss_k, gauss_k), 0)
    blurred = np.uint8(blurred)
    edges = cv2.Canny(blurred, low_th, high_th)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    # edges = cv2.erode(edges, np.ones((3,3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(
        edges, rho, theta, min_vote, np.array([]), min_line_length, max_line_gap
    )
    quad_lines = []
    true_lines = []
    slopes = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]
            m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
            atol_p = atol_p
            atol_m = atol_m
            for point in points:
                if (
                    np.isclose([x1, y1], point, atol=atol_p).all()
                    or np.isclose([x2, y2], point, atol=atol_p).all()
                ) and np.isclose(m1, sl, atol=atol_m).any():
                    quad_lines.append([x1, y1, x2, y2])

            if [x1, y1, x2, y2] not in quad_lines:
                true_lines.append([x1, y1, x2, y2])
                slopes.append([m1])

        for line in true_lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

    return img, slopes


def write_slope(slopes):
    with open("./slopes.txt", "w") as file:
        for item in slopes:
            # write each item on a new line
            file.write(f"{item[0]:.4f}\n")


def plot_image(image):
    plt.figure(figsize=(10, 7))
    plt.imshow(image, cmap="gray")
    plt.show()


def main(args):

    model_file = [f for f in os.listdir(args.model_path) if f.endswith(".h5")]
    print(model_file)
    if len(model_file) != 1:
        raise ValueError("there should be a model in there!")

    file_name = model_file[0]

    model = load_model((args.model_path + "/" + file_name), compile=False)

    # list images in image_dir
    files = os.listdir(args.image_dir)
    # print(files)
    # sys.stdout.flush()
    # randomly pick an image
    num = np.random.randint(1, len(files))
    # preprocess
    print(args.image_dir + "/" + files[num])
    sys.stdout.flush()

    input_image = preprocess(args.image_dir + "/" + files[num])
    # run inference
    prediction = (model.predict(input_image) > 0.5).astype(np.uint8)
    # reshape prediction
    prediction = np.reshape(prediction, (768, 1024))
    input_image = np.reshape(input_image, (768, 1024))

    ###############################################
    # find corners of triangle in mask
    corners = find_corners(prediction)
    # find quad corners
    p1, p2, p3 = edge_quad_points(corners, H=args.H)
    # mask regions of interest
    maskimage = region_of_interest(p1, p2, p3, input_image)
    # detect lines, return slopes and image with lines
    img, slopes = detect_lines(
        maskimage,
        p1,
        p2,
        p3,
        gauss_k=5,
        low_th=1,
        high_th=100,
        min_vote=50,
        min_line_length=30,
        max_line_gap=15,
        atol_p=30,
        atol_m=5,
    )
    # write slopes to file
    write_slope(slopes)
    # plot the image with lines
    plot_image(img)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="slip line args")

    parser.add_argument(
        "--image_dir",
        type=str,
        help="path to images",
        default=str(os.environ["WORK"]) + "/images_collective",
    )
    parser.add_argument(
        "--model_path", type=str, help="path to model", required=True
    )
    parser.add_argument(
        "--H", type=int, help="height of quads on edges", default=100
    )

    args = parser.parse_args()
    main(args)


# args = parser.parse_args()

# main(args)

