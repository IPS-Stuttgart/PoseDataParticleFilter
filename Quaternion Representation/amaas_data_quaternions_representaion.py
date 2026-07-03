import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import quaternion
import os

input_path= r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\PoseDataParticleFilter\Subject_1_F_1_poses.npz"
output_path= r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\PoseDataParticleFilter\Subject_1_F_1_poses_quaternion.npz"

# Create output folder if it doesn't exist
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#Get All files in the input folder
# all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.npz')]
# print(f"Found {len(all_files)} files to process")

# Load your .npz file
# for idx, filename in enumerate(all_files):
#    input_path  = os.path.join(INPUT_FOLDER, filename)
#    output_path = os.path.join(OUTPUT_FOLDER, filename.replace('.npz', '_quaternions.npz'))

#    print(f"\n[{idx+1}/{len(all_files)}] Processing: {filename}")

try:
      data = np.load(input_path, allow_pickle=True)
        
      trans         = data['trans']          
      gender        = str(data['gender'])    
      framerate     = float(data['mocap_framerate'])  
      betas         = data['betas']          # [16]
      dmpls         = data['dmpls']          # [T, 8]
      poses         = data['poses']          # [T, 156]

      T = poses.shape[0]

      # ── SMPL-H joint layout (156 = 52 joints × 3) ──────────────────────────────
      # poses[:, 0:3]    → root orientation       (1 joint)
      # poses[:, 3:66]   → body joints            (21 joints)
      # poses[:, 66:156] → hand joints            (30 joints, 15 per hand)

      NUM_JOINTS = 52  # SMPL-H total


      # ── Reshape to [T, 52, 3] axis-angle ───────────────────────────────────────
      poses_aa = poses.reshape(T, NUM_JOINTS, 3)   # axis-angle per joint
      Quaternion_arr = quaternion.from_rotation_vector(poses_aa)
      quaternion_poses_arr = quaternion.as_float_array(Quaternion_arr)

      # Verify
      norms = np.linalg.norm(quaternion_poses_arr, axis=-1)
      assert np.allclose(norms, 1.0), "Quaternion norms are not 1.0!"

      np.savez(
            output_path,
            trans           = data['trans'],
            betas           = data['betas'],
            gender          = data['gender'],
            dmpls           = data['dmpls'],
            mocap_framerate = data['mocap_framerate'],
            poses_quat      = quaternion_poses_arr,      # (T, 52, 4) [w, x, y, z]
        )
      print(f"Saved  | frames={T} | shape={quaternion_poses_arr.shape} | path={output_path}")
      

except Exception as e:
        print(f"FAILED: Subject_1_F_1_poses_quaternion | Error: {e}")
        raise
   


# convert axis angle to quaternion for one frame [Not in Use]

      # # print(poses.shape)
      # poses_aa = poses.reshape(T, NUM_JOINTS, 3)   # axis-angle per joint
      # # print(poses_aa.shape)
      # # temp_vect= poses_aa[0]
      # # frame_a= [temp_vect[0][0], temp_vect[0][1], temp_vect[0][2]]
      # Quaternion_arr = quaternion.from_rotation_vector(poses_aa)
      # quaternion_poses_arr = quaternion.as_float_array(Quaternion_arr)
      # # print(poses_quat_arr.shape)


# print(betas)
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