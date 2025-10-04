import cv2
import numpy as np

from engine3d.parse_poses import parse_poses

# from function.Blob_func import send_blob # 検知対象者保存用のために使用

# 定数定義
STRIDE = 8
BODY_3D_EDGES = np.array(
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

BODY_2D_EDGES = np.array(
    [[0, 1],  # neck - nose
     [1, 16], [16, 18],  # nose - l_eye - l_ear
     [1, 15], [15, 17],  # nose - r_eye - r_ear
     [0, 3], [3, 4], [4, 5],     # neck - l_shoulder - l_elbow - l_wrist
     [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
     [0, 6], [6, 7], [7, 8],        # neck - l_hip - l_knee - l_ankle
     [0, 12], [12, 13], [13, 14]]   # neck - r_hip - r_knee - r_ankle
     )

class PoseEstimator:
    def __init__(self, model, R, T, infer_request, input_tensor_name,
                 plotter, canvas_3d, canvas3d_window_name):
        self._model = model
        self._R = R
        self._t = T
        self._infer_request = infer_request
        self._input_tensor_name = input_tensor_name
        self._plotter = plotter
        self._canvas_3d = canvas_3d
        self._canvas3d_window_name = canvas3d_window_name
        self._current_time = cv2.getTickCount()
        self._mean_time = 0
        self._fx = -1

    def preprocessed(self, frame, size, flip, interpolation):
        self._frame = frame
        if size is not None:
            self._frame = cv2.resize(self._frame, size, interpolation=interpolation)
        if flip:
            self._frame = cv2.flip(self._frame, 1)

    def predict(self, before_3d_frame=None):
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
            poses_3d = _rotate_poses(poses_3d, self._R, self._t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = (-z, x, y)
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]

            # 転倒危険者を判定
            danger_person_index = _submit_joint(poses_3d, before_3d_frame)
            if danger_person_index == []:
                pass
            else:
                _draw_dangerous_person(self._frame, poses_2d, scaled_img, danger_person_index)
                # send_blob(self._frame)

            edges = (BODY_3D_EDGES + 19 *
                     np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape(-1, 2)

        self._plotter.plot(self._canvas_3d, poses_3d, edges)
        _draw_poses(self._frame, poses_2d, scaled_img)
        return self._frame, poses_3d

def _draw_dangerous_person(frame, poses_2d, scaled_img, danger_person_index):
    for index in danger_person_index:
        pose = np.array(poses_2d[index][0: -1]).reshape((-1, 3)).transpose()
        pose[0], pose[1] = (
            pose[0] * frame.shape[1] / scaled_img.shape[1],
            pose[1] * frame.shape[0] / scaled_img.shape[0]
        )

        max_pose_x = int(pose[0][pose[0] > -1.4].max())
        max_pose_y = int(pose[1][pose[1] > -1.4].max())
        min_pose_x = int(pose[0][pose[0] > -1.4].min())
        min_pose_y = int(pose[1][pose[1] > -1.4].min())

        cv2.rectangle(
            frame,
            (min_pose_x - 30, min_pose_y - 30),
            (max_pose_x + 30, max_pose_y + 30),
            (0, 0, 255),
            3,
            cv2.LINE_4
        )

        cv2.putText(
            frame,
            f"person{index + 1}",
            (min_pose_x - 40, min_pose_y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2
        )

def _draw_poses(frame, poses_2d, scaled_img):
    for pose in poses_2d:
        pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
        was_found = pose[2] > 0
        pose[0], pose[1] = (
            # 正確にジョイントをプロットするために座標の位置を調整する
            pose[0] * frame.shape[1] / scaled_img.shape[1],
            pose[1] * frame.shape[0] / scaled_img.shape[0]
        )

        for edge in BODY_2D_EDGES:
            if was_found[edge[0]] and was_found[edge[1]]:
                cv2.line(frame, tuple(pose[0:2, edge[0]].astype(np.int32)),
                        tuple(pose[0:2, edge[1]].astype(np.int32)),
                        (255, 255, 0),
                        4,
                        cv2.LINE_AA)

        for kpt_id in range(pose.shape[1]):
            if pose[2, kpt_id] != -1:
                cv2.circle(
                    frame,
                    tuple(pose[0:2, kpt_id].astype(np.int32)),
                    3,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )

def _rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)
    return poses_3d

def _submit_joint(poses_3d, before_poses_3d):
    index = 0
    danger_person_index = []

    # 初回フレームのみ有効
    if before_poses_3d is None:
        return danger_person_index
    else:
        try:
            for x in range(len(poses_3d)):
                if len(poses_3d) != len(before_poses_3d):
                    raise Exception("人数が変わりました")

                submit = np.abs(before_poses_3d[x] - poses_3d[x])
                submit = np.sum(submit)
                np.set_printoptions(precision=1, suppress=True)
                submit_total = int(submit)
                if submit_total >= 5000:
                    pass
                elif submit_total >= 300:
                    danger_person_index.append(index)
                    index += 1
            return danger_person_index
        except Exception as e:
            print(e)
            return danger_person_index
