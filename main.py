import cv2
import mediapipe as mp
import math
from typing import Tuple, Union

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
arr = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 32, 36, 37, 38, 39, 40, 43, 45, 50, 57, 58, 72, 73, 82, 83, 84, 85, 86, 87,
       90, 91, 92, 93, 106, 123, 131, 132, 135, 136, 137, 138, 140, 146, 147, 148, 149, 150, 152, 164, 165, 167, 169,
       170, 171, 172, 175, 176, 177, 178, 179, 180, 181, 182, 186, 187, 192, 194, 199, 200, 201, 202, 203, 204, 205,
       206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 220, 227, 234, 262, 266, 267, 268, 269, 270, 271, 273, 275,
       279, 280, 287, 287, 288, 302, 303, 304, 311, 312, 313, 314, 315, 316, 317, 322, 323, 326, 327, 335, 345, 352,
       361, 364, 365, 366, 367, 369, 376, 377, 378, 379, 391, 393, 394, 395, 396, 397, 397, 400, 401, 402, 403, 404,
       405, 406, 410, 411, 416, 418, 421, 422, 423, 424, 425, 426, 427, 428, 430, 431, 432, 433, 434, 435, 436, 447,
       454, 460]

drawing_spec = mp_drawing.DrawingSpec(thickness=0.1, circle_radius=0.01)

labels_dict = {0: 'without mask', 1: 'mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def draw_point(image, location, text):
    x = location.x
    y = location.y
    image_rows, image_cols, _ = image.shape
    # col = list(location.relative_keypoints)[feature]
    # row = list(location.relative_keypoints)[feature]
    keypoint_px = _normalized_to_pixel_coordinates(x, y, image_cols, image_rows)
    cv2.circle(image, keypoint_px, 2, (0, 0, 255), 2)
    # cv2.putText(image, str(text), keypoint_px, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return keypoint_px


def draw_rect(detection):
    location = detection.location_data
    image_rows, image_cols, _ = image.shape
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin, relative_bounding_box.ymin,
                                                        image_cols, image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin + relative_bounding_box.width,
                                                      relative_bounding_box.ymin + relative_bounding_box.height,
                                                      image_cols, image_rows)
    cv2.rectangle(image, rect_start_point, rect_end_point, (255, 255, 0), 10)
    return rect_start_point, rect_end_point


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int,image_height: int) -> Union[None, Tuple[int, int]]:
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils




webcam = cv2.VideoCapture(0)
#webcam = cv2.VideoCapture('http://172.20.10.3:8080/video')

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while webcam.isOpened():
        success, image = webcam.read()
        if not success:
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                try:
                    (x, y), (w, h) = draw_rect(detection)
                except:
                    (x, y), (w, h) = ((0, 0), (image.shape[0], image.shape[1]))
                crop_img = image[y:h, x:w]
                crop_img = increase_brightness(crop_img, value=90)
                with mp_face_mesh.FaceMesh(
                        max_num_faces=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as face_mesh:
                    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    annotated_image = image.copy()
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mouth_features = mouth_cascade.detectMultiScale(crop_img, 1.1, 20)
                            if len(mouth_features) > 0:
                                cv2.putText(image, labels_dict[0], (x + 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                            color_dict[0], 3)
                            else:
                                cv2.putText(image, labels_dict[1], (x + 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                            color_dict[1], 3)
        image = ResizeWithAspectRatio(image, width=1280)
        cv2.imshow("face detection", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
# webcam.release()