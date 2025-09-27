import cv2
import numpy as np
from engine3d.draw import draw_poses, draw_dangerous_person
from engine3d.process import rotate_poses, submit_joint
from engine3d.parse_poses import parse_poses
# from function.Blob_func import send_blob # 検知対象者保存用のために使用

# 定数定義
STRIDE = 8 
SKELETON_EDGES = np.array(
    [
        [0, 1],
        [0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
        [0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
        [1, 15], [15, 16],            # nose - l_eye - l_ear
        [1, 17], [17, 18],            # nose - r_eye - r_ear
        [0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
        [0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
    ]
)

class MultiThread:
    def __init__(self, frame, model, R, T, infer_request, input_tensor_name,
                 plotter, canvas_3d, canvas3d_window_name, current_time):
        self._frame = frame
        self._model = model
        self._R = R
        self._t = T
        self._infer_request = infer_request
        self._input_tensor_name = input_tensor_name
        self._plotter = plotter
        self._canvas_3d = canvas_3d
        self._canvas3d_window_name = canvas3d_window_name
        self._current_time = current_time
        self._mean_time = 0
        self._fx = -1

    def run(self, before_3d_frame=None):
        edges = []

        scaled_img = cv2.resize(self._frame, dsize=(self._model.inputs[0].shape[3], self._model.inputs[0].shape[2]))
        img = scaled_img[
            0: scaled_img.shape[0] - (scaled_img.shape[0] % STRIDE),
            0: scaled_img.shape[1] - (scaled_img.shape[1] % STRIDE)
        ]
        self._img = np.transpose(img, (2, 0, 1))[None]

        if self._fx < 0:
            self._fx = np.float32(0.8 * self._frame.shape[1])
        self._infer_request.infer({self._input_tensor_name: self._img})

        results = {
            name: self._infer_request.get_tensor(name).data[:]
            for name in {"features", "heatmaps", "pafs"}
        }
        results = (results["features"][0], results["heatmaps"][0], results["pafs"][0])
        poses_3d, poses_2d = parse_poses(results, 1, STRIDE, self._fx, is_video=True)

        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, self._R, self._t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = (-z, x, y)
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]

            # 転倒危険者を判定
            danger_person_index = submit_joint(poses_3d, before_3d_frame)
            if danger_person_index == []:
                pass
            else:
                draw_dangerous_person(self._frame, poses_2d, scaled_img, danger_person_index)
                # send_blob(self._frame)

            edges = (SKELETON_EDGES + 19 *
                     np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape(-1, 2)

        self._plotter.plot(self._canvas_3d, poses_3d, edges)
        draw_poses(self._frame, poses_2d, scaled_img)
        current_time = (cv2.getTickCount() - self._current_time) / cv2.getTickFrequency()

        if self._mean_time == 0:
            self._mean_time = current_time
        else:
            self._mean_time = self._mean_time * 0.95 + current_time * 0.05
        cv2.putText(self._frame, f"FPS: {int(1 / self._mean_time * 10) / 10}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))
        return self._frame, poses_3d
