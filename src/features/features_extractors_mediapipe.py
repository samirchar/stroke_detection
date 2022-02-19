import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from skimage import transform

torch.manual_seed(0)


def h_point_to_line_intersection(point, line):
    p1, p2 = line
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = point

    m = (y2 - y1) / (x2 - x1)
    b = y2 - m * x2

    iy = y3
    ix = (y3 - b) / m

    return ix, iy


def line_to_coors(line):
    x_coords = [i[0] for i in line]
    y_coords = [i[1] for i in line]
    return x_coords, y_coords


def eye_mouth_diag_angles(processed_landmarks):

    lm, rm = (processed_landmarks["mouth"][48], processed_landmarks["mouth"][54])

    le, re = (
        list(processed_landmarks["left_eye_center"].values())[0],
        list(processed_landmarks["right_eye_center"].values())[0],
    )

    eye_mouth_diag_angles = np.arccos(
        ((le - rm) * (re - lm))
        / (np.linalg.norm((le - rm)) * np.linalg.norm((re - lm)))
    )
    eye_mouth_diag_angles = np.degrees(eye_mouth_diag_angles)

    return eye_mouth_diag_angles


def nose_mouth_features(processed_landmarks):
    # TODO: Normalize distances by face max x distance and max y distance

    # Get landmarks
    x1, y1 = processed_landmarks["nose"][30]
    lm, rm = (processed_landmarks["mouth"][48], processed_landmarks["mouth"][54])

    x2, y2 = lm
    x3, y3 = rm

    # nose to mouth left corner
    lmnx = x1 - x2
    lmny = y2 - y1
    lmnd = np.sqrt(lmnx ** 2 + lmny ** 2)
    theta = np.degrees(np.arctan(lmny / lmnx))

    # nose to mouth right corner
    rmnx = x3 - x1
    rmny = y3 - y1
    rmnd = np.sqrt(rmnx ** 2 + rmny ** 2)
    phi = np.degrees(np.arctan(rmny / rmnx))
    return lmnx, lmny, lmnd, theta, rmnx, rmny, rmnd, phi


def line_horizontal_deviation(line):

    p1, p2 = line
    x1, y1 = p1
    x2, y2 = p2

    dev = np.degrees(np.arctan(np.abs((y2 - y1) / (x2 - x1))))

    return dev


def h_eyebrow_eyebrow_deviation(processed_landmarks, exclude=[]):

    """
    landmarks indices must be in ascending order from left to right
    and right eyebrow indices larger than left eyebrow
    """

    leb_filtered = {
        k: v for k, v in processed_landmarks["left_eyebrow"].items() if k not in exclude
    }
    reb_filtered = {
        k: v
        for k, v in processed_landmarks["right_eyebrow"].items()
        if k not in exclude
    }

    leb_filtered_sorted_idxs = np.argsort(list(leb_filtered.keys()))
    reb_filtered_sorted_idxs = np.argsort(list(reb_filtered.keys()))[::-1]

    leb_lmks = np.array(list(leb_filtered.values()))[leb_filtered_sorted_idxs]
    reb_lmks = np.array(list(reb_filtered.values()))[reb_filtered_sorted_idxs]

    deviations = []
    for leb, reb in zip(leb_lmks, reb_lmks):

        dev = line_horizontal_deviation([(leb), (reb)])
        deviations.append(dev)

    return deviations


# TODO: include mouth corners horizontal deviation?


class ROIExtractor:
    def __init__(self, frame, processed_landmarks, debug=False):
        self.frame = frame
        self.processed_landmarks = processed_landmarks
        self.debug = debug

    def get_region_of_interest(self, point1, point2, project_to=[]):

        """
        project_to = array of bbox vertices ordered starting at top left 
        and continue clockwise
        """

        min_x, max_x = min(point1[0], point2[0]), max(point1[0], point2[0])
        min_y, max_y = min(point1[1], point2[1]), max(point1[1], point2[1])
        roi_bounds = (min_x, max_x, min_y, max_y)

        if project_to != []:

            src = np.array(
                [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            )

            dst = np.array(project_to)
            dw, dh = dst.max(axis=0) - dst.min(axis=0)

            # dst_2 = dst_2*2 #because image sizes are not the same.
            tform = transform.estimate_transform("projective", src, dst)
            self.frame = transform.warp(self.frame, tform.inverse)

            roi = self.frame[min_y:max_y, min_x:max_x, :]
            roi = cv2.resize(roi, (dh, dw))

        else:
            roi = self.frame[min_y:max_y, min_x:max_x, :]

        if self.debug:
            plt.imshow(roi)
            plt.show()

        return roi, roi_bounds

    def left_mouth(self, **kwargs):
        tr = self.processed_landmarks["nose"][33]
        bl = np.array(
            (
                self.processed_landmarks["head"][2][0],
                self.processed_landmarks["head"][8][1],
            )
        )
        return self.get_region_of_interest(tr, bl, **kwargs)

    def right_mouth(self, **kwargs):
        p1 = self.processed_landmarks["nose"][33]
        p2 = np.array(
            (
                self.processed_landmarks["head"][14][0],
                self.processed_landmarks["head"][8][1],
            )
        )

        return self.get_region_of_interest(p1, p2, **kwargs)


class BarycentricOperations:
    def __init__(self, landmarks_left_to_right_mapping, triangle_indices, constant=100):

        self.landmarks_left_to_right_mapping = dict(landmarks_left_to_right_mapping)
        self.triangle_indices = (
            triangle_indices  # for dlip landmarks: triangle_indices = [17,26,30]
        )

        self.constant = constant

    def order_landmarks(self, landmarks):
        ordered_idxs = list(self.landmarks_left_to_right_mapping.keys()) + list(
            self.landmarks_left_to_right_mapping.values()
        )

        # Place triangle indices to the end
        for i in set(ordered_idxs) & set(self.triangle_indices):
            ordered_idxs.pop(ordered_idxs.index(i))

        ordered_idxs += self.triangle_indices

        return landmarks[ordered_idxs, :]

    def cartesian_to_barycentric(self, coordinates):

        """
        the last 3 coordinates of the array must be the reference triangle coordinates.
        
        """

        m, n = coordinates.shape

        homogeneous_coordinates = np.hstack(
            [coordinates, np.ones((m, 1)) * self.constant]
        )

        self.triangle = homogeneous_coordinates[-3:, :]

        barycentric_coordinates = homogeneous_coordinates[:-3, :] @ np.linalg.inv(
            self.triangle
        )

        return barycentric_coordinates

    def get_barycentric_landmarks(self, landmarks, order=True):

        # print(landmarks.shape)
        if order:
            ordered_landmarks = self.order_landmarks(landmarks)

        # print(ordered_landmarks.shape)
        return self.cartesian_to_barycentric(ordered_landmarks)

    def landmarks_true_horizontal_flip(self, ordered_landmarks):
        m, n = ordered_landmarks.shape
        c = m // 2
        flip_matrix = np.block(
            [[np.zeros((c, c)), np.eye(c, c)], [np.eye(c, c), np.zeros((c, c))]]
        )

        # print(flip_matrix.shape,ordered_landmarks.shape)
        ordered_landmarks_real_flip = flip_matrix @ ordered_landmarks

        return ordered_landmarks_real_flip

    def landmarks_theoretical_horizontal_flip(self, ordered_landmarks):
        return ordered_landmarks @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    def global_asymmetry_index(self, landmarks):

        barycentric_landmarks = self.get_barycentric_landmarks(landmarks)

        true_flipped_landmarks = self.landmarks_true_horizontal_flip(
            barycentric_landmarks
        )
        theoretical_flipped_landmarks = self.landmarks_theoretical_horizontal_flip(
            barycentric_landmarks
        )

        asymmetry_by_landmark = true_flipped_landmarks - theoretical_flipped_landmarks
        return (asymmetry_by_landmark ** 2).sum()


class OpticalFlow:
    def __init__(self, flow_threshold=6):
        self.flow_threshold = flow_threshold
        self.mask = None

    def compute_dense_flow(self, prev_frame, frame):

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        if self.mask is None:
            self.mask = np.zeros_like(prev_frame)

            # Sets image saturation to maximum
            self.mask[..., 1] = 255

        return magnitude, angle

    def filter_flow_magnitude_noise(self, magnitude):

        # Filter noise in magnitude.
        mean_flow_mag = magnitude.mean()
        magnitude[
            magnitude < self.flow_threshold * magnitude.mean()
        ] = 0  # TODO: This should be a hparam

        return magnitude

    def get_rgb_optical_flow(self, magnitude, angle):

        # Sets image hue according to the optical flow
        # direction
        self.mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        self.mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(self.mask, cv2.COLOR_HSV2BGR)

        self.mask = None

        return rgb


class OpticalFlowSymmetry:
    def __init__(self):
        pass

    def movement_score(self, magnitudes: list):

        magnitudes = np.array(magnitudes)
        return magnitudes.mean()

    def __symmetry_score(
        self, left_movement_score, right_movement_score, lambda_=3.8, ret="abs_diff"
    ):

        abs_dif = abs(left_movement_score - right_movement_score)

        raw_symmetry_score = 1 - lambda_ * abs_dif

        symmetry_score = np.clip(raw_symmetry_score, 0, 1)

        if ret == "abs_diff":
            return abs_dif
        elif ret == "raw_score":
            return raw_symmetry_score
        elif ret == "score":
            return symmetry_score
        else:
            raise ValueError("return type does not exit")

    def symmetry_score(
        self, left_magnitudes, right_magnitudes, lambda_=3.8, ret="abs_diff"
    ):

        left_movement_score = self.movement_score(left_magnitudes)
        right_movement_score = self.movement_score(right_magnitudes)

        return self.__symmetry_score(
            left_movement_score, right_movement_score, lambda_=lambda_, ret=ret
        )
