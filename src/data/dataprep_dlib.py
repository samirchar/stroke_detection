import torch
import cv2
import dlib
from collections import defaultdict
import imutils
from imutils import face_utils
from imutils.face_utils import shape_to_np
import numpy as np
from .imgutils import is_tensor_and_convert, rgb_to_bgr, img_2_gray_scale
import time
# from openface import openface

torch.manual_seed(0)


def filter_landmark_by_index(selected_landmarks, all_landmarks):

    filtered_landmarks = defaultdict(lambda: {"idxs": None})

    for k, v in all_landmarks.items():

        filtered_idxs = [j for j in v["idxs"] if j in selected_landmarks]

        if filtered_idxs != []:
            filtered_landmarks[k]["idxs"] = filtered_idxs
    return filtered_landmarks


def filter_landmark_by_group(selected_landmarks, all_landmarks):

    return {k: v for k, v in all_landmarks.items() if k in selected_landmarks}


def process_landmarks(landmarks, selected_landmarks=-1):

    # TODO: This should be a parameter in case landmarks change
    all_landmarks = {
        "mouth": {"idxs": list(range(48, 68))},
        "nose": {"idxs": list(range(27, 36))},
        "head": {"idxs": list(range(17))},
        "left_eye": {"idxs": [36, 37, 38, 39, 40, 41]},
        "left_eyebrow": {"idxs": [17, 18, 19, 20, 21]},
        "right_eye": {"idxs": [42, 43, 44, 45, 46, 47]},
        "right_eyebrow": {"idxs": [22, 23, 24, 25, 26]},
    }

    raw_num_landmarks = sum(map(lambda x: len(x["idxs"]), all_landmarks.values()))
    processed_num_landmarks = int(raw_num_landmarks)

    # Filter landmarks. TODO: This could be a method itself.
    if selected_landmarks == -1:
        filtered_landmarks = all_landmarks.copy()

    elif isinstance(selected_landmarks, list):
        if all(map(lambda x: isinstance(x, str), selected_landmarks)):
            filtered_landmarks = filter_landmark_by_group(
                selected_landmarks, all_landmarks
            )

        elif all(map(lambda x: isinstance(x, int), selected_landmarks)):
            filtered_landmarks = filter_landmark_by_index(
                selected_landmarks, all_landmarks
            )
        else:
            raise TypeError(
                "Selected landmarks must be all str, all int or an empty list"
            )
    else:
        raise TypeError("selected_landmarks can be either -1 or a list")

    processed_landmarks = defaultdict(lambda: {"idxs": None, "values": None})

    for k, v in filtered_landmarks.items():
        processed_landmarks[k] = dict(zip(v["idxs"], landmarks[v["idxs"]]))

    # Create new landmarks based on original ones. TODO: This could be a method itself.
    if "left_eye" in processed_landmarks.keys():
        processed_num_landmarks += 1
        left_eye = np.array([*processed_landmarks["left_eye"].values()])
        processed_landmarks["left_eye_center"] = {
            processed_num_landmarks - 1: np.mean(left_eye, axis=0).round().astype(int)
        }

    if "right_eye" in processed_landmarks.keys():
        processed_num_landmarks += 1
        right_eye = np.array([*processed_landmarks["right_eye"].values()])
        processed_landmarks["right_eye_center"] = {
            processed_num_landmarks - 1: np.mean(right_eye, axis=0).round().astype(int)
        }

    processed_num_landmarks += 1

    if "head" in processed_landmarks.keys():
        processed_num_landmarks += 1
        chin = np.array([*processed_landmarks["head"].values()])[6:11, :]

        processed_landmarks["chin"] = {
            processed_num_landmarks - 1: np.mean(chin, axis=0).round().astype(int)
        }

    processed_num_landmarks += 1

    return processed_landmarks


def draw_landmarks(face, img_, gray_img_, selected_landmarks, landmark_radius=1):
    """
    img must be in BGR format
    """

    img = img_.copy()
    gray_img = gray_img_.copy()

    for group_landmarks in face["landmarks"].values():
        for (x, y) in group_landmarks.values():
            cv2.circle(img, (x, y), landmark_radius, (0, 0, 255), -1)

    # print(processed_landmarks)
    return img


def get_sorted_dict_values_by_key(dictionary, reverse=False):

    return [dictionary[key] for key in sorted(dictionary.keys(), reverse=reverse)]


def processed_landmark_dict_to_array(processed_landmarks):

    flattened_landmark_dict = {
        k: v for i in processed_landmarks.values() for k, v in i.items()
    }

    flattened_sorted_landmark_array = np.vstack(
        get_sorted_dict_values_by_key(flattened_landmark_dict)
    )

    return flattened_sorted_landmark_array

class DNNDetector:
    
    def __init__(self,
                 model_file="models/landmarks/res10_300x300_ssd_iter_140000.caffemodel",
                 config_file="models/landmarks/deploy.prototxt.txt"):
        
        self.model_file = model_file
        self.config_file = config_file
        self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        
    def detect(self,img):
        '''
        Only returns one bbox for the most probable face
        '''
        
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 117.0, 123.0))
        self.net.setInput(blob)
        faces = self.net.forward()
        #to draw faces on image
        max_confidence = 0
        box = None
        for i in range(faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence > max_confidence:
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    max_confidence = max(max_confidence,confidence)

        rect = dlib.rectangle(x, y, x1, y1)
        
        return rect,max_confidence

class ProcessImage:
    def __init__(self, predictor, detector, shape_predictor):

        self.predictor = predictor
        self.detector = detector
        self.shape_predictor = shape_predictor
        self.time_face_detector = 0
        self.time_landmark_detector = 0

    def __detector_wrapper(self,detector,img,gray_img):
        if isinstance(self.detector,DNNDetector):
            rect,prob = self.detector.detect(img)
        else: #Default to HOG face detector
            rect = self.detector(gray_img, 1)[0]
        return rect
    def preprocess_image(self, img_, width=800, is_bgr=False):

        # Width sets a fix img width.

        img = is_tensor_and_convert(img_)

        if not is_bgr:
            img = rgb_to_bgr(img)

        if width is not None:
            img = imutils.resize(img, width=width)

        gray_img = img_2_gray_scale(img)

        return img, gray_img

    def crop_and_align(self, img_, gray_img_, method="face_chip", size=128):

        img = img_.copy()
        gray_img = gray_img_.copy()

        # TODO: Confirm when to use gray img

        tic_face_detector = time.time()

        rect = self.__detector_wrapper(self.detector,img,gray_img)

        toc_face_detector = time.time()
        self.time_face_detector += toc_face_detector-tic_face_detector

        faces = dlib.full_object_detections()

        tic_landmarks = time.time()

        faces.append(self.predictor(gray_img, rect))

        toc_landmarks = time.time()
        self.time_landmark_detector += toc_landmarks-tic_landmarks

        if len(faces) == 0:
            raise ValueError("No face detected")
        if len(faces) > 1:
            raise ValueError("only supports one face")

        if method == "face_chip":
            face_aligned = dlib.get_face_chip(img, faces[0], size=size)

            # Recalculate landmarks instead of transforming
            face_aligned_gray = img_2_gray_scale(face_aligned)
            #rects = self.detector(face_aligned_gray, 1)
            rect2 = self.__detector_wrapper(self.detector,face_aligned,face_aligned_gray)
            landmarks_aligned = self.predictor(face_aligned_gray, rect2)
            landmarks_aligned = shape_to_np(landmarks_aligned)

        elif method == "openface":
        
            '''
            face_aligner = openface.AlignDlib(self.shape_predictor)

            # Use openface to calculate and perform the face alignment
            face_aligned = face_aligner.align(
                size,
                img,
                rects[0],
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE,
            )

            # Recalculate landmarks instead of transforming
            face_aligned_gray = img_2_gray_scale(face_aligned)
            rects = self.detector(face_aligned_gray, 1)
            landmarks_aligned = self.predictor(face_aligned_gray, rects[0])
            landmarks_aligned = shape_to_np(landmarks_aligned)
            '''

            pass
        elif method == "face_aligner":
            fa = FaceAligner(self.predictor, desiredFaceWidth=size)

            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = imutils.face_utils.rect_to_bb(rect)
            face_aligned, landmarks_aligned = fa.align(img, gray_img, rect)

        elif method is None:
            face_aligned = img.copy()
            landmarks_aligned = self.predictor(gray_img, rect)
            landmarks_aligned = shape_to_np(landmarks_aligned)
        else:
            raise ValueError(f"method: {method} not supported")

        face_aligned_gray = img_2_gray_scale(face_aligned)

        return rect,face_aligned, face_aligned_gray, landmarks_aligned

    def process(
        self,
        img,
        width=800,
        is_bgr=False,
        method="face_chip",
        size=128,
        selected_landmarks=-1,
    ):

        img, gray_img = self.preprocess_image(img, width, is_bgr)
        rect, img, gray_img, landmarks_aligned = self.crop_and_align(
            img, gray_img, method, size
        )

        processed_landmarks = process_landmarks(landmarks_aligned, selected_landmarks)

        # Left as dict b/c we might want to store more info about the face
        face = {"landmarks": processed_landmarks}

        return rect,img, gray_img, face


FACIAL_LANDMARKS_IDXS = face_utils.FACIAL_LANDMARKS_IDXS
FACIAL_LANDMARKS_68_IDXS = face_utils.FACIAL_LANDMARKS_68_IDXS
FACIAL_LANDMARKS_5_IDXS = face_utils.FACIAL_LANDMARKS_5_IDXS


class FaceAligner:
    def __init__(
        self,
        predictor,
        desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256,
        desiredFaceHeight=None,
    ):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # simple hack ;)
        if len(shape) == 68:
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = desiredRightEyeX - self.desiredLeftEye[0]
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (
            (leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2,
        )

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(
            (int(eyesCenter[0]), int(eyesCenter[1])), angle, scale
        )

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += tX - eyesCenter[0]
        M[1, 2] += tY - eyesCenter[1]

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        aligned_landmarks = np.squeeze(cv2.transform(np.expand_dims(shape, 1), M), 1)

        # TODO: aligned bounding box

        # return the aligned face
        return aligned_face, aligned_landmarks
