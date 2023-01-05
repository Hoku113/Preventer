### Human Pose Estimation 3D demo from OpenCV
import cv2
import time
import collections
import threading
import json
from openvino.runtime import Core
from function.process_function import *
# from function.multi_thread import FrameProcess3D
from humanpose3d.draw import Plotter3d, draw_poses
from humanpose3d.process import rotate_poses
from humanpose3d.parse_poses import parse_poses
from function.utils import VideoPlayer

focal_length = -1  # default
stride = 8
player = None
skeleton_set = None
processing_times = []
mean_time = 0
fx = -1

MODEL_PATH = "./model/intel/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml"
MODEL_WEIGHTS_PATH = "./model/intel/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin"
FILE_PATH = "./data/extrinsics.json"

ie_core = Core()

model = ie_core.read_model(model=MODEL_PATH, weights=MODEL_WEIGHTS_PATH)

compiled_model = ie_core.compile_model(model=model, device_name="CPU")
infer_request = compiled_model.create_infer_request()
input_tensor_name = model.inputs[0].get_any_name()
input_layer = compiled_model.input(0)
output_layers = list(compiled_model.outputs)

canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
plotter = Plotter3d(canvas_3d.shape[:2])
canvas_3d_window_name = "Canvas 3D"
cv2.namedWindow(canvas_3d_window_name)
cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

if FILE_PATH is None:
    print("no such file or directory")

with open(FILE_PATH, "r") as f:
    extrinsics = json.load(f)
R = np.array(extrinsics['R'], dtype=np.float32)
t = np.array(extrinsics['t'], dtype=np.float32)

video_player = VideoPlayer(0, flip=True, fps=30, skip_first_frames=0)
video_player.start()

# input_image = video_player.next()

while cv2.waitKey(1) != 27:
    current_time = cv2.getTickCount()
    frame = video_player.next()
    
    if frame is None:
        break

    input_scale = 450 / frame.shape[0]
    scaled_img = cv2.resize(frame, dsize=(model.inputs[0].shape[3], model.inputs[0].shape[2]))
    img = scaled_img[
        0: scaled_img.shape[0] - (scaled_img.shape[0] % stride),
        0: scaled_img.shape[1] - (scaled_img.shape[1] % stride),
    ]

    img = np.transpose(img, (2, 0, 1))[None]

    if fx < 0:
        fx = np.float32(0.8 * frame.shape[1])

    infer_request.infer({input_tensor_name: img})
    results = {
        name: infer_request.get_tensor(name).data[:]
        for name in {"features", "heatmaps", "pafs"}
    }

    results = (results["features"][0], results["heatmaps"][0], results["pafs"][0])

    poses_3d, poses_2d = parse_poses(results, input_scale, stride, fx)
    edges = []

    if len(poses_3d):
        poses_3d = rotate_poses(poses_3d, R, t)
        poses_3d_copy = poses_3d.copy()
        x = poses_3d_copy[:, 0::4]
        y = poses_3d_copy[:, 1::4]
        z = poses_3d_copy[:, 2::4]

        poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, y
        poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
        edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape(-1, 2)
    plotter.plot(canvas_3d, poses_3d, edges)
    cv2.imshow(canvas_3d_window_name, canvas_3d)

    draw_poses(frame, poses_2d)
    current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
    if mean_time == 0:
        mean_time = current_time
    else:
        mean_time = mean_time * 0.95 + current_time * 0.05
    
    cv2.putText(frame, f"FPS: {int(1 / mean_time * 10) / 10,}", (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    cv2.imshow("ICV 3D Human Pose Estimation", frame)

cv2.destroyAllWindows()
    

# resize_scale = 450 / input_image.shape[1]
# WIDTH = int(input_image.shape[1] * resize_scale)
# HEIGHT = int(input_image.shape[0] * resize_scale)

# engine3D = engine.Engine3js(grid=True, axis=True, view_width=WIDTH, view_height=HEIGHT)

# imgbox = widgets.Image(
#     format=".jpg", height=HEIGHT, width=WIDTH
# )
# display(widgets.HBox([engine3D.renderer, imgbox]))

# skeleton = engine.Skeleton(body_edges=body_edges)

# processing_times = collections.deque()

# while cv2.waitKey(1) != 27:
#     frame = video_player.next()

#     if frame is None:
#         print("frame does not exist!!")
#         break

#     if threading.activeCount() == 2:
#         th = FrameProcess3D(frame, model, compiled_model, infer_request, input_tensor_name, engine3D, skeleton, imgbox, focal_length, stride, processing_times)
#         frame = th.run()

#     cv2.imshow('streaming', frame)

#     engine3D.renderer.render(engine3D.scene, engine3D.cam)

#     if skeleton_set:
#         engine3D.scene_remove(skeleton_set)

# video_player.stop()
# cv2.destroyAllWindows()

