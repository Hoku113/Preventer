# Human Pose Estimation 3D demo from OpenCV
import argparse
import json
import time

import cv2
import numpy as np
import openvino as ov
import yt_dlp

from engine3d.draw import Plotter3d
from engine3d.pose_estimator import PoseEstimator

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

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print("Error opening video stream or file")
        return -1

    capture.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
    interpolation = (
        cv2.INTER_AREA
        if size[0] < capture.get(cv2.CAP_PROP_FRAME_WIDTH) else cv2.INTER_LINEAR
    )

    # 姿勢推定する人
    pose_estimator = PoseEstimator(model, R, T, infer_request, input_tensor_name,
                                   plotter, canvas_3d, canvas_3d_window_name)

    first_time = True
    while cv2.waitKey(1) != ESC_KEY:
        start_time = time.time()
        ret, frame = capture.read()
        if not ret:
            print("End of stream")
            break

        # フレームの前処理
        pose_estimator.preprocessed(frame, size, flip, interpolation)

        # 3D姿勢推定
        if first_time:
            frame, before_3d_frame = pose_estimator.predict()
            first_time = False
        else:
            frame, before_3d_frame = pose_estimator.predict(before_3d_frame)

        # FPS計算
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 3d Human pose
        cv2.imshow(canvas_3d_window_name, canvas_3d)

        # 2d Human pose
        cv2.imshow("ICV 3D Human Pose Estimation", frame)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        type=str,
                        default="0",
                        help="string. default: your webcam \n video path: local video \n video URL: youtube video")
    parser.add_argument('--size',
                        type=int,
                        default=(720, 720),
                        help="Image size")
    parser.add_argument('--flip',
                        type=bool,
                        default=False,
                        help="bool. Flipped images")
    parser.add_argument('--skip_first_frames',
                        type=int,
                        default=0,
                        help="int. Skip frames when load video or video URL")
    args, unparsed = parser.parse_known_args()
    main(args.source, args.size, args.flip, args.skip_first_frames)
