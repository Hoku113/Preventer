import cv2
import numpy as np
from engine3d.draw import Plotter3d, draw_poses, draw_dangerous_person
from engine3d.process import rotate_poses, submit_joint
from engine3d.parse_poses import parse_poses
from function.debug_function import debug_function
from function.Blob_func import send_blob

class MultiThread:

    def __init__(self, frame, model, stride, fx, R, t, infer_request, input_tensor_name,
                 plotter, canvas_3d, canvas3d_window_name, current_time, mean_time):
        self._frame = frame
        self._model = model
        self._stride = stride
        self._fx = fx
        self._R = R
        self._t = t
        self._infer_request = infer_request
        self._input_tensor_name = input_tensor_name
        self._plotter = plotter
        self._canvas_3d = canvas_3d
        self._canvas3d_window_name = canvas3d_window_name
        self._current_time = current_time
        self._mean_time = mean_time


    def run(self, before_3d_frame=None):

        scaled_img = cv2.resize(self._frame, dsize=(self._model.inputs[0].shape[3], self._model.inputs[0].shape[2]))

        img = scaled_img[
            0: scaled_img.shape[0] - (scaled_img.shape[0] % self._stride),
            0: scaled_img.shape[1] - (scaled_img.shape[1] % self._stride)
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
        poses_3d, poses_2d = parse_poses(results, 1, self._stride, self._fx, is_video=True)

        edges = []

        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, self._R, self._t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = (
                -z,
                x,
                -y
            )

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]

            danger_person_index = submit_joint(poses_3d, before_3d_frame)

            if danger_person_index == None or danger_person_index == []:
                pass
            elif danger_person_index:
                draw_dangerous_person(self._frame, poses_2d, scaled_img, danger_person_index)
                # send_blob(self._frame)

            edges = (Plotter3d.SKELETON_EDGES + 19 *
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
