import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(0)


def is_tensor_and_convert(img):
    assert isinstance(img, torch.Tensor) | isinstance(
        img, np.ndarray
    ), "image format not supported"

    if isinstance(img, torch.Tensor):
        img = img.numpy()

    return img


def img_2_gray_scale(img):
    """
    img must be in BGR
    """
    img = is_tensor_and_convert(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_image_local(img):
    """
    img must be in BGR format
    """
    # show the output img with the face detections + facial landmarks
    # cv2.imshow("Output", img)
    plt.imshow(bgr_to_rgb(img))
    plt.show()


class annotate_data:
    def __init__(self, initial_h=50, interval_h=30):
        self.initial_h = initial_h
        self.interval_h = interval_h
        self.current_max_h = 0

    def annotate(self, frame_drawn, name, data, font=cv2.FONT_HERSHEY_SIMPLEX):

        self.current_max_h += self.interval_h

        frame_drawn = cv2.putText(
            frame_drawn,
            f"{name}: {str(data)}",
            (50, self.current_max_h),
            font,
            self.interval_h / 50,
            (0, 0, 255),
            2,
            cv2.LINE_4,
        )
        return frame_drawn


def destroy_all_cam_windows(cap):
    cap.release()
    cv2.destroyAllWindows()
    for i in range(1, 5):
        cv2.waitKey(1)
