import numpy as np

data = np.load("helpers/box_init_pose.npy")

np.savetxt("box_init_pose.txt", data)
