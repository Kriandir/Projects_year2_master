import numpy as np


u_list = np.random.rand(10)
v_list = np.random.rand(10)
for u,v in np.nditer([u_list,v_list], op_flags=["readwrite"]):
    print u
    u[...] = u*v
    v[...] = 2
