### Human Pose Estimation demo from OpenCV
import cv2
import time
import collections
from openvino.runtime import Core
from function.process_function import *
from function.config import colors, default_skeleton

POSE_MODEL_PATH = './model/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml'
WEIGHT_POSE_MODEL_PATH = './model/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin'

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

# get cv2 video frame from web cam
cap = cv2.VideoCapture(0)

while cv2.waitKey(1)!=27:

    ret, frame = cap.read()

    if frame is None:
        print("source ended")
        break

    scale = 1280 / max(frame.shape)
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale,interpolation=cv2.INTER_AREA)

    
    input_img = cv2.resize(frame, (HEIGHT, WIDTH), cv2.INTER_AREA)

    input_img = input_img.transpose((2, 1, 0))[np.newaxis, ...]

    start_time = time.perf_counter()
    results = compiled_model([input_img])
    stop_time = time.perf_counter()

    pafs = results[pafs_output_key]
    heatmaps = results[heatmaps_output_key]

    poses, scores = process_results(frame, compiled_model, pafs, heatmaps)

    frame = draw_poses(frame, poses, 0.1)

    cv2.imshow('streaming', frame)

    processing_times.append(stop_time - start_time)
    if len(processing_times) > 200:
        processing_times.popleft()
    
    _, f_width = frame.shape[:2]
    processing_time = np.mean(processing_times) * 1000
    fps = 1000 / processing_time

    cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)


    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
