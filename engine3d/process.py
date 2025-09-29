import numpy as np


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)
    return poses_3d

def submit_joint(poses_3d, before_poses_3d):
    index = 0
    danger_person_index = []

    # 初回フレームのみ有効
    if before_poses_3d is None:
        return danger_person_index
    else:
        try:
            for x in range(len(poses_3d)):
                if len(poses_3d) != len(before_poses_3d):
                    raise Exception("人数が変わりました")

                submit = np.abs(before_poses_3d[x] - poses_3d[x])
                submit = np.sum(submit)
                np.set_printoptions(precision=1, suppress=True)
                submit_total = int(submit)
                if submit_total >= 5000:
                    pass
                elif submit_total >= 300:
                    danger_person_index.append(index)
                    index += 1
            return danger_person_index
        except Exception as e:
            print(e)
            return danger_person_index
