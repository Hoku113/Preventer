### Human Pose Estimation 3D demo from OpenCV
import cv2
import time
import collections
import threading
from openvino.runtime import Core
from function.process_function import *
from function.multi_thread import FrameProcess3D
from function.utils import VideoPlayer
import function.engine.engine3js as engine
import ipywidgets as widgets
from IPython.display import clear_output, display
from function.engine.parse_poses import parse_poses
from function.process_function import *
from function.config import body_edges

focal_length = -1  # default
stride = 8
player = None
skeleton_set = None
processing_times = []

MODEL_PATH = "./model/intel/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml"
MODEL_WEIGHTS_PATH = "./model/intel/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin"

ie_core = Core()

model = ie_core.read_model(model=MODEL_PATH, weights=MODEL_WEIGHTS_PATH)

compiled_model = ie_core.compile_model(model=model, device_name="CPU")
infer_request = compiled_model.create_infer_request()
input_tensor_name = model.inputs[0].get_any_name()
input_layer = compiled_model.input(0)
output_layers = list(compiled_model.outputs)

video_player = VideoPlayer(0, flip=True, fps=30, skip_first_frames=0)
video_player.start()

input_image = video_player.next()

resize_scale = 450 / input_image.shape[1]
WIDTH = int(input_image.shape[1] * resize_scale)
HEIGHT = int(input_image.shape[0] * resize_scale)

engine3D = engine.Engine3js(grid=True, axis=True, view_width=WIDTH, view_height=HEIGHT)

imgbox = widgets.Image(
    format=".jpg", height=HEIGHT, width=WIDTH
)
display(widgets.HBox([engine3D.renderer, imgbox]))

skeleton = engine.Skeleton(body_edges=body_edges)

processing_times = collections.deque()

while cv2.waitKey(1) != 27:
    frame = video_player.next()

    if frame is None:
        print("frame does not exist!!")
        break

    if threading.activeCount() == 2:
        th = FrameProcess3D(frame, model, compiled_model, infer_request, input_tensor_name, engine3D, skeleton, imgbox, focal_length, stride, processing_times)
        frame = th.run()

    cv2.imshow('streaming', frame)

    engine3D.renderer.render(engine3D.scene, engine3D.cam)

    if skeleton_set:
        engine3D.scene_remove(skeleton_set)

video_player.stop()
cv2.destroyAllWindows()

