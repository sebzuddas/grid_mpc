""" A version of main.py but with no libary, pure linear algebra"""

# A = np.array(
    #     [[-D/M, 1/M, 0, 1/M],
    #     [0, -1/T_t, 1/T_t, 0], 
    #     [-1/(R*T_g), 0, -1/T_g, 0], 
    #     [0, 0, 0, -1/T_dr]]) #System Matrix

    # B = np.array(
    #     [[0, 0],
    #     [0, 0],
    #     [1/T_g, 0],
    #     [0, 1/T_dr]]) # Input Matrix

    # E = np.array(
    #     [[-1/M],
    #     [0],
    #     [0],
    #     [0]]) # Disturbances
    
    # C = np.array([50, 0, 0, 0]) # Output matrix

    # x = np.array([[1], [1], [0], [1]])# system starting states, dw(t), dpm(t), dpv(t), dpdr(t)
    # u = np.array([[0], [0]]) # system inputs, dpm,ref(t) dpdr,ref(t)
