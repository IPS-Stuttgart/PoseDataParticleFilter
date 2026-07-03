import numpy as np

# data = np.load(":\Users\RaghavBansal\OneDrive - ProcDNA Analytics Pvt. Ltd\Desktop\amass\KIT\KIT\3\912_3_01_poses.npz")
# import numpy as np

data = np.load(r"C:\Users\ragha\Desktop\important ids and documents\ml research prof.florian\PoseDataParticleFilter\Subject_1_F_1_poses.npz")
print(data.files)   # shows all arrays inside


# # Access a specific array
arr = data["poses"]  
# print(len(arr[0]))
# for i in range(len(arr)):
#     for j in range(len(arr[0])):
#         print(arr[i][j], end= " ")

#     print('')
print(arr)


# from scipy.spatial.transform import Rotation as R

# Load your .npz file
# data = np.load('your_kit_sequence.npz', allow_pickle=True)

# trans         = data['trans']          # [T, 3]
# gender        = str(data['gender'])    # 'male'/'female'/'neutral'
# framerate     = float(data['mocap_framerate'])  # e.g. 120.0
# betas         = data['betas']          # [16]
# dmpls         = data['dmpls']          # [T, 8]
# poses         = data['poses'] 