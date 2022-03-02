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
import math as m
import mediapipe as mp
from src.data.videosource import WebcamSource
from src.data.custom.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

# from openface import openface

torch.manual_seed(0)

#Helper function that extracts the indexes of mediapipe region of interests (e.g. mouth landmarks)

def extract_roi_indexes(roi_name):
    mp_roi = getattr(mp.solutions.face_mesh,roi_name)
    return list(set([j for i in list(mp_roi) for j in i]))

###Face Landmarks region of interest to Indices##
#Left and right indexes has been mirrored such that left = actual persons right

POINTS_IDX = [33, 263, 61, 291, 199] #Key points to detect head pose

RIGHT_EYE_IDXS = extract_roi_indexes('FACEMESH_LEFT_EYE')
LEFT_EYE_IDXS = extract_roi_indexes('FACEMESH_RIGHT_EYE')

LEFT_IRIS_IDXS = extract_roi_indexes('FACEMESH_RIGHT_IRIS')
RIGHT_IRIS_IDXS = extract_roi_indexes('FACEMESH_LEFT_IRIS')

RIGHT_EYEBROW_IDXS = extract_roi_indexes('FACEMESH_LEFT_EYEBROW')
LEFT_EYEBROW_IDXS = extract_roi_indexes('FACEMESH_RIGHT_EYEBROW')

HEAD_IDXS = extract_roi_indexes('FACEMESH_FACE_OVAL')
MOUTH_IDXS = extract_roi_indexes('FACEMESH_LIPS')

ADDITIONAL_MOUTH_IDXS = [38,
                        41,
                        42,
                        72,
                        73,
                        74,
                        77,
                        85,
                        86,
                        89,
                        90,
                        96,
                        179,
                        180,
                        183,
                        184,
                        268,
                        271,
                        272,
                        302,
                        303,
                        304,
                        307,
                        315,
                        316,
                        319,
                        320,
                        325,
                        403,
                        404,
                        408,
                        407]
NOSE_IDXS = [4]

LEFT_UPPER_EAR_IDXS = [127]
RIGHT_UPPER_EAR_IDXS = [356]

BOTTOM_CHIN_IDXS = [152]
MID_CHIN_IDXS = [200]

KEY_LANDMARKS = {
    "mouth": {"idxs": MOUTH_IDXS+ADDITIONAL_MOUTH_IDXS},
    "nose": {"idxs": NOSE_IDXS},
    "head": {"idxs": HEAD_IDXS},
    "mid_chin": {"idxs": MID_CHIN_IDXS},
    "left_eye": {"idxs": LEFT_EYE_IDXS},
    "left_eyebrow": {"idxs": LEFT_EYEBROW_IDXS},
    "right_eye": {"idxs": RIGHT_EYE_IDXS},
    "right_eyebrow": {"idxs": RIGHT_EYEBROW_IDXS}
}

##FaceMesh Configuration##
STATIC_MODE=False
MAX_FACES=1
MIN_DETECTION_CONFIDENCE=0.5
MIN_TRACK_CONFIDENCE=0.5
REFINE_LANDMARKS=True
FRAME_HEIGHT=720
FRRAME_WIDTH=1280
CHANNELS=3
DESIRED_FACE_WIDTH_HEIGHT = (128,128)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)


#3D rotation matrices  
def Rx(theta):
    theta = np.radians(theta)
    return np.array([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
    theta = np.radians(theta)
    return np.array([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
    theta = np.radians(theta)
    return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def landmark_2d_normalization(landmarks_3D,
                              pose,
                              transform = True,
                              normalize = True,
                              center = True,
                              draw = False):

    SIZE = 100
    
    points=landmarks_3D.copy()
    
    #Pose corrrection matrix
    R= Rx(pose[0])@Ry(-pose[1])@Rz(pose[2])

    #Correct pose
    if transform:
        t_points = (R@points.T).T
    else:
        t_points = points
    
    #Get only de 2d landmarks
    t_points_2d = t_points[:,:2]
    
    #Normalize between 0 and 1
    if normalize:
        #t_points_2d = (t_points_2d-t_points_2d.min())/(t_points_2d.max()-t_points_2d.min())
        
        leftEyePts = t_points_2d[LEFT_EYE_IDXS,:]
        rightEyePts = t_points_2d[RIGHT_EYE_IDXS,:]
        
        leftEyeCenter = leftEyePts.mean(axis=0)
        rightEyeCenter = rightEyePts.mean(axis=0)

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        
        desiredDist = 0.2

        scale = desiredDist / dist
        
        t_points_2d = t_points_2d*scale

    t_points_2d = t_points_2d*np.array([SIZE,SIZE])
    
    if center:
        nose = t_points_2d[4]
        t_points_2d=t_points_2d-nose+np.array([SIZE//2,SIZE//2]) 
        
    if draw:
        img = np.zeros((SIZE,SIZE,3),dtype=np.uint8)
        img.fill(255)

        for (x, y) in t_points_2d:
            cv2.circle(img,
                       (int(round(x)),int(round(y))),
                       SIZE//256,
                       (0, 0, 255),
                       -1)
        cv2.imshow("Image",img)
    return t_points_2d

def filter_landmark_by_index(selected_landmarks, key_landmarks):
    filtered_landmarks = defaultdict(lambda: {"idxs": None})

    for k, v in key_landmarks.items():

        filtered_idxs = [j for j in v["idxs"] if j in selected_landmarks]

        if filtered_idxs != []:
            filtered_landmarks[k]["idxs"] = filtered_idxs
    return filtered_landmarks


def filter_landmark_by_group(selected_landmarks, key_landmarks):
    return {k: v for k, v in key_landmarks.items() if k in selected_landmarks}

def process_landmarks(landmarks,selected_landmarks=-1, key_landmarks = KEY_LANDMARKS ):

    raw_num_landmarks = sum(map(lambda x: len(x["idxs"]), key_landmarks.values()))
    processed_num_landmarks = int(raw_num_landmarks)

    # Filter landmarks. TODO: This could be a method itself.
    if selected_landmarks == -1:
        filtered_landmarks = key_landmarks.copy()

    elif isinstance(selected_landmarks, list):
        if all(map(lambda x: isinstance(x, str), selected_landmarks)):
            filtered_landmarks = filter_landmark_by_group(
                selected_landmarks, key_landmarks
            )

        elif all(map(lambda x: isinstance(x, int), selected_landmarks)):
            filtered_landmarks = filter_landmark_by_index(
                selected_landmarks, key_landmarks
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
        left_eye = np.array([*processed_landmarks["left_eye"].values()])
        processed_landmarks["left_eye_center"] = {
            processed_num_landmarks: np.mean(left_eye, axis=0)
        }
        processed_num_landmarks += 1

    if "right_eye" in processed_landmarks.keys():
        right_eye = np.array([*processed_landmarks["right_eye"].values()])
        processed_landmarks["right_eye_center"] = {
            processed_num_landmarks: np.mean(right_eye, axis=0)
        }
        processed_num_landmarks += 1

    return processed_landmarks

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


class FaceAligner:
    def __init__(
        self,
        landmarks,
        head_pose = [None,None,None],
        desiredLeftEye=(0.35, 0.35),
        desiredFaceWidthHeight = DESIRED_FACE_WIDTH_HEIGHT,
        left_eye_idxs = LEFT_EYE_IDXS,#[133],#[127,162,67,140,103,109],
        right_eye_idxs = RIGHT_EYE_IDXS,#[362],#[356,389,297,378,338,332],
        yaw_correction = True
    ):
        # store the facial landmarks, desired output left
        # eye position, and desired output face width + height
        
        self.left_eye_idxs = left_eye_idxs
        self.right_eye_idxs = right_eye_idxs
        self.landmarks = landmarks
        self.desiredLeftEye = desiredLeftEye
        self.head_pose = head_pose #as x,y,z = pitch, yaw, roll
        self.pitch, self.yaw, self.roll = self.head_pose
        self.desiredFaceWidth, self.desiredFaceHeight = desiredFaceWidthHeight
        self.yaw_correction = yaw_correction
        
    def align(self, image):
        # convert the landmark (x, y)-coordinates to a NumPy array
        # compute the center of mass for each eye
        
        leftEyePts = self.landmarks[self.left_eye_idxs,:]
        rightEyePts = self.landmarks[self.right_eye_idxs,:]
        
        leftEyeCenter = leftEyePts.mean(axis=0).round().astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).round().astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        
        if self.roll is None:
            self.roll = np.degrees(np.arctan2(dY, dX)) - 180#TODO: Not working

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))

        if self.yaw_correction:
            dist = abs(dist/np.cos(np.radians(self.yaw)))#Yaw correction
        
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
            (int(eyesCenter[0]), int(eyesCenter[1])), -self.roll, scale
        )

        #print((int(eyesCenter[0]), int(eyesCenter[1])), -self.roll, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += tX - eyesCenter[0]
        M[1, 2] += tY - eyesCenter[1]

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        aligned_landmarks = np.squeeze(cv2.transform(np.expand_dims(self.landmarks, 1), M), 1)

        return aligned_face, aligned_landmarks


def get_camara_matrix(frame_width,frame_height):
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]],
        dtype="double",
    )
    return camera_matrix

class FaceMeshDetector():
 
    def __init__(self,
                 staticMode=STATIC_MODE,
                 maxFaces=MAX_FACES,
                 minDetectionCon=MIN_DETECTION_CONFIDENCE,
                 minTrackCon=MIN_TRACK_CONFIDENCE,
                 refine_landmarks=REFINE_LANDMARKS,
                 frame_height=FRAME_HEIGHT,
                 frame_width=FRRAME_WIDTH,
                 desiredFaceWidthHeight = DESIRED_FACE_WIDTH_HEIGHT,
                 channels=CHANNELS,
                 points_idx = POINTS_IDX,
                 yaw_correction=True,
                 roll_as_eyes_angle = False):
     
        self.yaw_correction = yaw_correction
        self.roll_as_eyes_angle = roll_as_eyes_angle
        self.desiredFaceWidthHeight = desiredFaceWidthHeight
        self.refine_landmarks = refine_landmarks
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.channels = channels
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.points_idx = points_idx
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,
                                                 self.maxFaces,
                                                 min_detection_confidence = self.minDetectionCon,
                                                 min_tracking_confidence = self.minTrackCon,
                                                 refine_landmarks=self.refine_landmarks)
        
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
 
        self.source = WebcamSource()
    
        self.camera_matrix = get_camara_matrix(self.frame_width,
                                               self.frame_height)
        self.pcf = PCF(
                    near=1,
                    far=10000,
                    frame_height=self.frame_height,
                    frame_width=self.frame_width,
                    fy=self.camera_matrix[1, 1],
                )
        
        self.points_idx = self.points_idx + [key for (key, val) in procrustes_landmark_basis]
        self.points_idx = list(set(self.points_idx))
        self.points_idx.sort()

    def get_2Dlandmarks_and_pose(self,landmarks_3D):
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_3D])
        landmarks = landmarks.T

        metric_landmarks, pose_transform_mat = get_metric_landmarks(
            landmarks.copy(), self.pcf
        )

        landmarks_2D = landmarks[0:2,:].T*np.array([self.frame_width, self.frame_height])
        landmarks_3D = landmarks.T

        image_points = landmarks_2D[self.points_idx,:][None, :]#The 2D points necessary for pose estimation
        model_points = metric_landmarks[0:3, self.points_idx].T#The 3D model points necessary for pose estimation

        # see here:
        # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
        pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
        mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat[:3, :3])
        mp_translation_vector = pose_transform_mat[:3, 3, None]

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(mp_rotation_vector)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the rotation degree. Right and up is positve (wrt the person, not image)
        x = np.sign(angles[0])*180-angles[0]
        y = angles[1] 
        z = -angles[2]
        
        return landmarks_2D, landmarks_3D, x, y, z
    
    def crop_and_align(self,img,landmarks,rotation_point,angle):
        rotation_point = landmarks[1]
        rotMat = cv2.getRotationMatrix2D(rotation_point,angle,1.0) #Get the rotation matrix, its of shape 2x3
        img_rotated = cv2.warpAffine(img,rotMat,img.shape[1::-1]) #Rotate the image
        landmarks_rotated = np.hstack([landmarks,np.ones((len(landmarks),1))]).T
        landmarks_rotated = np.dot(rotMat,landmarks_rotated).T #Perform Dot product and get back the points in shape of (4,2)

        return img_rotated, landmarks_rotated

    def findFaceMesh(self, img):
        self.imgRGB = img.copy()
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:

                face = {}
                landmarks_2D,landmarks_3D, x, y, z = self.get_2Dlandmarks_and_pose(faceLms.landmark[:468])
                
                #img, landmarks_2D = self.crop_and_align(img,landmarks_2D,landmarks_2D[1],-z)
                
                if (self.maxFaces==1):
                    fa = FaceAligner(landmarks_2D,
                                    head_pose = [x,
                                                 y,
                                                 None if self.roll_as_eyes_angle else z],
                                                 
                                    desiredFaceWidthHeight= self.desiredFaceWidthHeight,
                                    yaw_correction=self.yaw_correction)

                    self.imgRGB, landmarks_2D=fa.align(self.imgRGB)
                    
                face['landmarks'] = landmarks_2D
                face['landmarks_normalized'] = landmark_2d_normalization(landmarks_3D,[x,y,z])
                face['landmarks_3D'] = landmarks_3D
                face['pose'] = [x,y,z]
                face['detection'] = faceLms
                faces.append(face)
  
            #Align Roll.
            
            #cv2.putText(img, "x: " + str(np.round(x,2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #cv2.putText(img, "y: " + str(np.round(y,2)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            #cv2.putText(img, "z: " + str(np.round(z,2)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)

        return self.imgRGB, faces
    
    def draw_faces(self,img_,faces,draw_face_mesh = False,use_raw_landmarks = False):
        
        img = img_.copy()
        for face in faces:

            if use_raw_landmarks:
                landmarks = face['landmarks']
            else:
                landmarks = processed_landmark_dict_to_array(face['processed_landmarks'])

            if draw_face_mesh:
                self.mpDraw.draw_landmarks(img, face['detection'], self.mpFaceMesh.FACEMESH_CONTOURS,
                                       self.drawSpec, self.drawSpec)
            else:
                
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for (x, y) in landmarks:
                    cv2.circle(img,
                               (int(round(x)),int(round(y))),
                               self.desiredFaceWidthHeight[0]//256,
                               (0, 0, 255),
                               -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        return img

    def process(self,img,selected_landmarks=-1):
    
        img = is_tensor_and_convert(img)
        img, faces = self.findFaceMesh(img)
        
        for i in range(len(faces)):
            faces[i]["processed_landmarks"] = process_landmarks(faces[i]['landmarks'], selected_landmarks)
            faces[i]["processed_landmarks_normalized"] = process_landmarks(faces[i]['landmarks_normalized'], selected_landmarks)

        return img, faces

    def run_live_v2(self,save_faces_objs = False,draw_face_mesh=False):
        cap = cv2.VideoCapture(0)
        #ret, frame = cap.read()
        pTime = 0
        self.live_faces_objs = []
        while True:
            success, frame = cap.read()

            frame, faces = self.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),selected_landmarks=-1)
            frame = self.draw_faces(frame,faces,draw_face_mesh=draw_face_mesh)
            if save_faces_objs:
                self.live_faces_objs.append(faces)

            #if len(faces)!= 0:
            #    print(faces[0])
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, f'FPS: {int(fps)}', (5, 120), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), 1)
            cv2.imshow("Image", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            c = cv2.waitKey(1)
            if c == 27:
                break

    def run_live(self):
        cap = cv2.VideoCapture(0)
        #ret, frame = cap.read()
        pTime = 0

        while True:
            success, frame = cap.read()

            frame, faces = self.findFaceMesh(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = self.draw_faces(frame,faces,use_raw_landmarks=True)
            #if len(faces)!= 0:
            #    print(faces[0])
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, f'FPS: {int(fps)}', (5, 120), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), 1)
            cv2.imshow("Image", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            c = cv2.waitKey(1)
            if c == 27:
                break

