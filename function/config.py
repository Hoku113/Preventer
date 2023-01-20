import numpy as np

# HumanPose 
colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

# HumanPose 3D
# 3D edge index array
body_edges = np.array(
    [
        [0, 1], 
        [0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
        [0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
        [1, 15], [15, 16],            # nose - l_eye - l_ear
        [1, 17], [17, 18],            # nose - r_eye - r_ear
        [0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
        [0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
    ]
)


body_edges_2d = np.array(
    [
        [0, 1],                       # neck - nose
        [1, 16], [16, 18],            # nose - l_eye - l_ear
        [1, 15], [15, 17],            # nose - r_eye - r_ear
        [0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
        [0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
        [0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
        [0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
    ]
)  