# Human Pose Estimation 3D demo from OpenCV
import cv2
import threading
import json
import numpy as np
import yt_dlp
import argparse
import openvino as ov
from engine3d.draw import Plotter3d
from function.utils import VideoPlayer
from engine3d.thread import MultiThread

# model settings 
MODEL_PATH = "./model/intel/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml"
MODEL_WEIGHTS_PATH = "./model/intel/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin"
FILE_PATH = "./data/extrinsics.json"
ESC_KEY = 27

def main(source, size, flip, skip_first_frames):
    # モデルの初期化
    ie_core = ov.Core()
    model = ie_core.read_model(model=MODEL_PATH, weights=MODEL_WEIGHTS_PATH)
    compiled_model = ie_core.compile_model(model=model, device_name="CPU")
    infer_request = compiled_model.create_infer_request()
    input_tensor_name = model.inputs[0].get_any_name()

    # 3Dキャンバスの初期化
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = "Canvas 3D"
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, plotter.mouse_callback)

    if FILE_PATH is None:
        print("no such file or directory")
        return -1
    else:
        with open(FILE_PATH, "r") as f:
            extrinsics = json.load(f)
        R = np.array(extrinsics['R'], dtype=np.float32)
        T = np.array(extrinsics['t'], dtype=np.float32)

    if source.startswith("https"):
        with yt_dlp.YoutubeDL({"format": "bestvideo[ext=mp4][vcodec*=avc1]/best[ext=mp4]/best"}) as ydl:
            info_dict = ydl.extract_info(source, download=False)
            source = info_dict.get("url", None)
    elif source == "0":
        source = int(source)

    # web camera
    video_player = VideoPlayer(source, size, flip, skip_first_frames)
    video_player.start()

    first_time = True
    while cv2.waitKey(1) != ESC_KEY:
        current_time = cv2.getTickCount()
        frame = video_player.next()
        if frame is None:
            break

        if threading.active_count() == 3:
            th = MultiThread(frame, model, R, T, infer_request, input_tensor_name, 
                             plotter, canvas_3d, canvas_3d_window_name, current_time)

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
    parser.add_argument('--size', type=int, default=(720, 720) , help="Image size")
    parser.add_argument('--flip', type=bool, default=False, help="bool. Flipped images")
    parser.add_argument('--skip_first_frames', type=int, default=0, help="int. Skip frames when load video or video URL")
    args, unparsed = parser.parse_known_args()
    main(args.source, args.size, args.flip, args.skip_first_frames)