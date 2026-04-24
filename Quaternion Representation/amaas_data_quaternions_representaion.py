import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import quaternion


# Load your .npz file
# data = np.load('C:\Users\RaghavBansal\OneDrive - ProcDNA Analytics Pvt. Ltd\Desktop\amass\KIT\KIT\3\912_3_01_poses.npz', allow_pickle=True)

path = Path(r'C:\Users\RaghavBansal\OneDrive - ProcDNA Analytics Pvt. Ltd\Desktop\amass\KIT\KIT\3\912_3_01_poses.npz')
data = np.load(path, allow_pickle=True)

trans         = data['trans']          # [T, 3]
gender        = str(data['gender'])    # 'male'/'female'/'neutral'
framerate     = float(data['mocap_framerate'])  # e.g. 120.0
betas         = data['betas']          # [16]
dmpls         = data['dmpls']          # [T, 8]
poses         = data['poses']          # [T, 156]

T = poses.shape[0]
# print(T)
# ── SMPL-H joint layout (156 = 52 joints × 3) ──────────────────────────────
# poses[:, 0:3]    → root orientation       (1 joint)
# poses[:, 3:66]   → body joints            (21 joints)
# poses[:, 66:156] → hand joints            (30 joints, 15 per hand)

NUM_JOINTS = 52  # SMPL-H total


#  rotation_vector = [wx, wy, wz]

#Quaternion = quaternion.from_rotation_vector(rotation_vector)

 #angle = sqrt(wx **2 + wy**2 + wz**2)
 #axis:
    # ax = wx/angle ...
    # ay = wy/angle
    # az= wz/ angle

 # quaternion= (ax, ay, az , angle)

# ── Reshape to [T, 52, 3] axis-angle ───────────────────────────────────────
poses_aa = poses.reshape(T, NUM_JOINTS, 3)   # axis-angle per joint
temp_vect= poses_aa[0]
print(temp_vect)
# Quaternion = quaternion.from_rotation_vector(temp_vect)
# print(temp_vect)
# # print(poses_aa.shape)
# # ── Convert axis-angle → quaternion [T, 52, 4] ─────────────────────────────
# # scipy Rotation expects shape [N, 3] so flatten time & joints, then reshape back
# poses_flat = poses_aa.reshape(-1, 3)                        # [T*52, 3]
# rots = R.from_rotvec(poses_flat)                            # axis-angle = rotvec
# quats = rots.as_quat()                                      # [T*52, 4]  (x, y, z, w)
# poses_quat = quats.reshape(T, NUM_JOINTS, 4)                # [T, 52, 4]
# print(quats.shape)
# print(poses_quat)
# # ── Split into semantic groups ──────────────────────────────────────────────
# root_orient_quat  = poses_quat[:, 0:1,  :]   # [T,  1, 4] global orientation
# body_pose_quat    = poses_quat[:, 1:22, :]   # [T, 21, 4] body joints
# hand_pose_quat    = poses_quat[:, 22:,  :]   # [T, 30, 4] hand joints

# print("root_orient_quat :", root_orient_quat.shape)   # (T, 1,  4)
# print("body_pose_quat   :", body_pose_quat.shape)     # (T, 21, 4)
# print("hand_pose_quat   :", hand_pose_quat.shape)     # (T, 30, 4)
# print("trans            :", trans.shape)              # (T, 3)