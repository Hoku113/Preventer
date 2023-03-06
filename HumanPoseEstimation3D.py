# Human Pose Estimation 3D demo from OpenCV
import cv2
import threading
import json
import pafy
import argparse
from openvino.runtime import Core
from function.process_function import *
from engine3d.draw import Plotter3d
from function.utils import VideoPlayer
from engine3d.thread import MultiThread

# model and 3d plot settings 

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

def main(source, size, flip, fps, skip_first_frames):

    first_time = True

    if source.startswith("https"):
        video=pafy.new(source)
        best = video.getbest(preftype="mp4")
        source = best.url
    elif source == "0":
        source = int(source)

    # web camera
    video_player = VideoPlayer(source, size, flip, fps, skip_first_frames)

    video_player.start()

    while cv2.waitKey(1) != 27:
        current_time = cv2.getTickCount()
        frame = video_player.next()
        
        if frame is None:
            break

        if threading.active_count() == 2:
            th = MultiThread(frame, model, stride, fx, R, t, infer_request,
                    input_tensor_name,  plotter, canvas_3d, canvas_3d_window_name, current_time, mean_time)

            if first_time:
                frame, before_3d_frame = th.run()
                first_time = False
            else:
                frame, before_3d_frame = th.run(before_3d_frame)
            

        # 3d Human pose
        cv2.imshow(canvas_3d_window_name, canvas_3d)

        # 2d Human pose
        cv2.imshow("ICV 3D Human Pose Estimation", frame)

    video_player.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="0", help="string. default: your webcam \n video path: local video \n video URL: youtube video")
    parser.add_argument('--size', type=int, default=None, help="Image size")
    parser.add_argument('--flip', type=bool, default=False, help="bool. Flipped images")
    parser.add_argument('--fps', type=int, default=30, help="int. Change frame rate")
    parser.add_argument('--skip_first_frames', type=int, default=0, help="int. Skip frames when load video or video URL")
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS.source, FLAGS.size, FLAGS.flip, FLAGS.fps, FLAGS.skip_first_frames)