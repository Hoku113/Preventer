import cv2
import threading
import time
import function.engine.engine3js as engine
import ipywidgets as widgets
from IPython.display import clear_output, display
from function.engine.parse_poses import parse_poses
from function.process_function import *
from function.config import body_edges


class FrameProcess(threading.Thread):

    def __init__(self, frame, compiled_model, pafs_output_key, heatmaps_output_key, processing_times, height, width, before_points=[]):
        super(FrameProcess, self).__init__()
        self._frame = frame
        self._compiled_model = compiled_model
        self._pafs_output_key = pafs_output_key
        self._heatmaps_output_key = heatmaps_output_key
        self._processing_times = processing_times
        self._height = height
        self._width = width
        self._before_points = before_points

    def run(self):

        self._frame_gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)

        scale = 1280 / max(self._frame_gray.shape)
        if scale < 1:
            self._frame = cv2.resize(self._frame_gray, None, fx=scale, interpolation=cv2.INTER_AREA)
            input_img = cv2.resize(self._frame, (self._height, self._width), cv2.INTER_AREA)
        else:
            input_img = cv2.resize(self._frame_gray, (self._width, self._height), cv2.INTER_AREA)

        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

        start_time = time.time()
        results = self._compiled_model([input_img])
        stop_time = time.time()

        pafs = results[self._pafs_output_key]
        heatmaps = results[self._heatmaps_output_key]

        poses, scores = process_results(self._frame, self._compiled_model, pafs, heatmaps)
        
        frame, before_points = draw_poses(self._frame, poses, 0.1, self._before_points)
        

        self._processing_times.append(stop_time - start_time)
        if len(self._processing_times) > 200:
            self._processing_times.popleft()

        _, f_width = frame.shape[:2]
        processing_time = np.mean(self._processing_times) * 1000
        fps = 1000 / processing_time

        cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)

        return frame, before_points

            

class FrameProcess3D(threading.Thread):

    def __init__(self, frame, model, compiled_model, infer_request, input_tensor_name, engine3D, skeleton, imgbox, forcal_length, stride, processing_times):
        self._frame = frame
        self._model = model
        self._compiled_model = compiled_model
        self._infer_request = infer_request
        self._input_tensor_name = input_tensor_name
        self._engine3D = engine3D
        self._skeleton = skeleton
        self._imgbox = imgbox
        self._forcal_length = forcal_length
        self._stride = stride
        self._processing_times = processing_times


    def run(self):
        # self._frame_gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)

        # resize_scale = 450 / self._frame_gray.shape[1]
        # WIDTH = int(self._frame_gray.shape[1] * resize_scale)
        # HEIGHT = int(self._frame_gray.shape[0] * resize_scale)

        # engine3D = engine.Engine3js(grid=True, axis=True, view_width=WIDTH, view_height=HEIGHT)

        # imgbox = widgets.Image(
        #     format=".jpg", height=HEIGHT, width=WIDTH
        # )
        # display(widgets.HBox([engine3D.renderer, imgbox]))

        # skeleton = engine.Skeleton(body_edges=body_edges)

        # inference

        scaled_img = cv2.resize(self._frame, dsize=(self._model.inputs[0].shape[3], self._model.inputs[0].shape[2]))

        if self._forcal_length < 0:
            focal_length = np.float32(0.8 * scaled_img.shape[1])

        start_time = time.time()

        inference_result = model_infer(scaled_img, self._stride, self._infer_request, self._input_tensor_name)

        stop_time = time.time()
        self._processing_times.append(stop_time - start_time)

        poses_3d, poses_2d = parse_poses(inference_result, 1, self._stride, focal_length, True)

        if len(self._processing_times) > 200:
            self._processing_times.popleft()

        processing_time = np.mean(self._processing_times) * 1000
        fps = 1000 / processing_time

        if len(poses_3d) > 0:

            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = (
                -z + np.ones(poses_3d[:, 2::4].shape) * 200,
                -y + np.ones(poses_3d[:, 2::4].shape) * 100,
                -x,
            )

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            people = self._skeleton(poses_3d=poses_3d)

            try:
                self._engine3D.scene_remove(skeleton_set)
            except Exception:
                pass

            self._engine3D.scene_add(people)
            skeleton_set = people

            # draw 2D
            frame = draw_poses(self._frame, poses_2d, scaled_img)

        else:
            try:
                self._engine3D.scene_remove(skeleton_set)
                skeleton_set = None
            except Exception:
                pass

        
        cv2.putText(
            self._frame,
            f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        self._imgbox.value = cv2.imencode(
            ".jpg",
            self._frame,
            params=[cv2.IMWRITE_JPEG_QUALITY, 90],
        )[1].tobytes()

        return self._frame