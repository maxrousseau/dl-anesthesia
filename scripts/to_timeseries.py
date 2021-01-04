import numpy as np


x = np.load("x.npy")
y = np.load("y.npy")

# import IPython ; IPython.embed() ; exit(1)

'''
ok so X is shape of (3740, 306) and i want to reshape to (3740, 6, 51)

'''

x_time = np.reshape(x, (3740, 6, 51), order='F')
np.save("x_time.npy", x_time)
