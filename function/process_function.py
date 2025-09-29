import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided

from function.decoder import OpenPoseDecoder

# 定数定義
COLORS = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))
DEFAULT_SKELTON = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling
    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """

    # padding
    A = np.pad(A, padding, mode="constant")
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
    )
    A_w = A_w.reshape(-1, *kernel_size)

    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)

# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)

# get poses from results
def process_results(img, compiled_model, pafs, heatmaps):
    # This processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py

    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
    )
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    decoder = OpenPoseDecoder()
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores

def draw_poses(img, poses, point_score_threshold, before_points, skeleton=DEFAULT_SKELTON):
    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]

        if before_points == []:
            pass
        elif before_points[before_points < 640].min() == 0:
            print("passed")
            pass
        else:
            print(f"subject points: {np.abs(points - before_points)}")

        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, COLORS[i], 2)

        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=COLORS[j], thickness=4)

    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    before_points = points
    return img, before_points

def model_infer(scaled_img, stride, infer_request, input_tensor_name):
    img = scaled_img[
        0 : scaled_img.shape[0] - (scaled_img.shape[0] % stride),
        0 : scaled_img.shape[1] - (scaled_img.shape[1] % stride),
    ]
    img = np.transpose(img, (2, 0, 1))[None]

    infer_request.infer({input_tensor_name: img})
    results = {
        name: infer_request.get_tensor(name).data[:]
        for name in {"features", "heatmaps", "pafs"}
    }
    results = (results["features"][0], results["heatmaps"][0], results["pafs"][0])

    return results

