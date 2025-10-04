import cv2
import numpy as np

from engine3d.one_euro_filter import OneEuroFilter


class Pose:
    def __init__(self, keypoints, confidence):
        self.keypoints = keypoints
        self.confidence = confidence
        self.id = None
        self.last_id = -1
        self.translation_filter = [OneEuroFilter(freq=80, beta=0.01),
                                   OneEuroFilter(freq=80, beta=0.01),
                                   OneEuroFilter(freq=80, beta=0.01)]
        found_keypoints = np.zeros((np.count_nonzero(self.keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(keypoints.shape[0]):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        self.bbox = cv2.boundingRect(found_keypoints)

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = self.last_id + 1
            self.last_id += 1

    def filter(self, translation):
        filtered_translation = []
        for coordinate_id in range(3):
            filtered_translation.append(self.translation_filter[coordinate_id](translation[coordinate_id]))
        return filtered_translation
