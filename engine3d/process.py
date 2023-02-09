import numpy as np

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


def submit_joint(poses_3d, before_poses_3d):
    try:
        submit = np.abs(before_poses_3d - poses_3d)
        submit = np.sum(submit)
        np.set_printoptions(precision=1, suppress=True)
        submit_total = int(submit)

        if submit_total >= 600:
            pass
        else:
            return submit_total
    except:
        pass