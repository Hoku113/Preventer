import cv2
import threading
import time
from function.process_function import *

class FrameProcess(threading.Thread):

    def __init__(self, frame, compiled_model, pafs_output_key, heatmaps_output_key, processing_times, height, width):
        super(FrameProcess, self).__init__()
        self._frame = frame
        self._compiled_model = compiled_model
        self._pafs_output_key = pafs_output_key
        self._heatmaps_output_key = heatmaps_output_key
        self._processing_times = processing_times
        self._height = height
        self._width = width

    def run(self):

        self._frame_gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)

        scale = 1280 / max(self._frame_gray.shape)
        if scale < 1:
            self._frame = cv2.resize(self._frame_gray, None, fx=scale, interpolation=cv2.INTER_AREA)
            input_img = cv2.resize(frame, (self._height, self._width), cv2.INTER_AREA)
        else:
            input_img = cv2.resize(self._frame, (self._width, self._height), cv2.INTER_AREA)

        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

        start_time = time.time()
        results = self._compiled_model([input_img])
        stop_time = time.time()

        pafs = results[self._pafs_output_key]
        heatmaps = results[self._heatmaps_output_key]

        poses, scores = process_results(self._frame, self._compiled_model, pafs, heatmaps)

        frame = draw_poses(self._frame, poses, 0.1)

        self._processing_times.append(stop_time - start_time)
        if len(self._processing_times) > 200:
            self._processing_times.popleft()

        _, f_width = frame.shape[:2]
        processing_time = np.mean(self._processing_times) * 1000
        fps = 1000 / processing_time

        cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)

        return frame

            


            

            