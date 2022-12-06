### Human Pose Estimation demo from OpenCV
import cv2
import time
import collections
import threading
from openvino.runtime import Core
from function.process_function import *
from function.multi_thread import FrameProcess
from function.utils import VideoPlayer

POSE_MODEL_PATH = './model/intel/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.xml'
WEIGHT_POSE_MODEL_PATH = './model/intel/human-pose-estimation-0001/FP16-INT8/human-pose-estimation-0001.bin'

# read model
ie_core = Core()
model = ie_core.read_model(model=POSE_MODEL_PATH, weights=WEIGHT_POSE_MODEL_PATH)
compiled_model = ie_core.compile_model(model=model, device_name="CPU")

pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")

input_layer = compiled_model.input(0)
output_layers = list(compiled_model.outputs)

HEIGHT, WIDTH = list(input_layer.shape)[2:]

processing_times = collections.deque() 

# video_player = VideoPlayer("https://github.com/intel-iot-devkit/sample-videos/blob/master/store-aisle-detection.mp4?raw=true", flip=True, fps=30, skip_first_frames=500)
video_player = VideoPlayer(0, flip=True, fps=30, skip_first_frames=500)

video_player.start()

processing_times = collections.deque()


while cv2.waitKey(1) != 27:

    frame = video_player.next()

    if frame is None:
        print("source ended")
        break

    if(threading.activeCount() == 2):
        th = FrameProcess(frame, compiled_model, pafs_output_key, heatmaps_output_key, processing_times, HEIGHT, WIDTH)
        frame = th.run()

# デバッグ用
#--------------------------------------------------------

    # scale = 1280 / max(frame.shape)
    # if scale < 1:
    #     frame = cv2.resize(frame, None, fx=scale,interpolation=cv2.INTER_AREA)

    # print(frame.shape)

    
    # input_img = cv2.resize(frame, (WIDTH, HEIGHT), cv2.INTER_AREA)

    # input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

    # start_time = time.perf_counter()
    # results = compiled_model([input_img])
    # stop_time = time.perf_counter()

    # pafs = results[pafs_output_key]
    # heatmaps = results[heatmaps_output_key]

    # poses, scores = process_results(frame, compiled_model, pafs, heatmaps)

    # print(poses)
    # print(scores)

    # frame = draw_poses(frame, poses, 0.1)

    # processing_times.append(stop_time - start_time)
    # if len(processing_times) > 200:
    #     processing_times.popleft()
    
    # _, f_width = frame.shape[:2]
    # processing_time = np.mean(processing_times) * 1000
    # fps = 1000 / processing_time

    # cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)

# --------------------------------------------------------

    cv2.imshow('streaming', frame)

video_player.stop()
cv2.destroyAllWindows()
