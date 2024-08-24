""" A version of main.py but with no libary, pure linear algebra"""

import numpy as np
import control as ct
import control.matlab
import matplotlib.pyplot as plt
import math


def main():


    Ts = 0.1 # sampling time

    N = 8 # prediction horizon

    #### Defining the Model #####

    # Model params
    M = 8
    D = 0.8 
    R = 0.05 
    T_t = 0.5 
    T_g = 0.2 
    T_dr = 0.25 # demand response time (s)


    Ac = np.array(
        [[-D/M, 1/M, 0, 1/M],
        [0, -1/T_t, 1/T_t, 0], 
        [-1/(R*T_g), 0, -1/T_g, 0], 
        [0, 0, 0, -1/T_dr]]) #System Matrix

    Bc = np.array(
        [[0, 0],
        [0, 0],
        [1/T_g, 0],
        [0, 1/T_dr]]) # Input Matrix

    Ec = np.array(
        [[-1/M],
        [0],
        [0],
        [0]]) # Disturbances
    
    Cc = np.array([[50], [0], [0], [0]]) # Output matrix
    Cc = Cc.T
    # print(C)
    # exit()

    # D = np.zeros(p, n)


    # Weighting Matrices
    #TODO: Check
    # Q = C @ C.T # C'*C - weights on the states
    # R = np.transpose(B) @ B # B'*B
    


    # print(B.shape[1])
    # print(Ac.shape, Bc.shape, Cc.shape)
    # print(Ac)
    # print(Bc)
    # print(Cc)
    
    
    n = Ac.shape[0] # number of states nxn
    m = Bc.shape[1] # number of inputs nxp
    p = Cc.shape[0] # 
    
    Dc = np.zeros((p, m))
    # print(Dc)
    

    #obtains the discrete time prediction model
    sysc = ct.matlab.ss(Ac, Bc, Cc, Dc)
    sysd = ct.matlab.c2d(sysc, Ts)
    A, B, C, D = ct.matlab.ssdata(sysd)
    # print(K[0])
    # print(K[1])
    # print(K[2])
    # print(K[3])
    # print(A)
    # print(B)
    # print(C)

    # Weighting matrices
    # in this case, it's better to have the weighting matrices as eye() since the other values result in sq matrices but with zero weighting. 
    Q = np.eye(n) # squared state
    R = np.eye(m) # squared number of inputs
    # print(Q)
    # print(R)
    
    
    K_inf = ct.dlqr(A, B, Q, R)
    K_2 = -1 * ct.place(A, B,[0.01, 0, 0, 0.01])
    # print(K_inf)
    # print(K_2)

    A_dlyap = (A+(B@K_2)).T
    Q_dlyap = (K_2.T @ R @ K_2) + Q
    
    # print(Q_dlyap)


    P = ct.dlyap(A_dlyap, Q_dlyap)
    # print(P)

    F, G = predict_mats(A, B, N) # where N is the prediction horizon

    H, L, M_cost = cost_mats(F, G, Q, R, P)

    # print(F, G, H, L, M_cost)

    S = np.linalg.solve(H, -L)
    # S becomes an m*N matrix (ie it is the set of optimal control inputs along the horizon, calculated in a receding manner)
    print(S)
    
    
    KN = S[0:m, :]
    
    # check for stability:
    spectral_radius(A, B, KN)
   
    
    

    x0 = np.array([[1], 
                   [1], 
                   [0], 
                   [1]])# system starting states, dw(t), dpm(t), dpv(t), dpdr(t)
    
    Uopt = S@x0

    u0 = np.array([[0], 
                   [0]]) # system inputs, dpm,ref(t) dpdr,ref(t)
    
    timesteps = 50
    k = 0
    x = x0

    xs = []
    us = []
    
    while k <= timesteps:

        Uopt = KN@x# calculate the optimal control sequence
        uopt = Uopt[0:n, :] # extract the first m from the sequence
        x = A @ x + B @ uopt
        k += 1
        xs.append(x)
        us.append(uopt)

    plot_output(xs, us)

    

def spectral_radius(A: np.array, B:np.array, KN:np.array):
     ### Calculating the stability of A+B@KN ###

    Z = A+B@KN
    eigenvals = np.linalg.eigvals(Z)
    sr = np.max(np.abs(eigenvals))
    
    print('Stabilising', sr) if sr < 1 else print('Not Stabilising', sr)

def plot_output(xs: list, us: list):
    # # Convert lists to numpy arrays for easier manipulation
    # expect xs, us to be a list of numpy arrays
    xs = np.array(xs).squeeze()
    us = np.array(us).squeeze()

    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Plotting states
    plt.subplot(2, 1, 1)
    for i in range(xs.shape[1]):
        plt.plot(xs[:, i], label=f'State {i+1}')
    plt.title('State Trajectories')
    plt.xlabel('Time Step')
    plt.ylabel('State Values')
    plt.legend()

    # Plotting control inputs
    plt.subplot(2, 1, 2)
    for i in range(us.shape[1]):
        plt.plot(us[:, i], label=f'Control {i+1}')
    plt.title('Control Inputs')
    plt.xlabel('Time Step')
    plt.ylabel('Control Values')
    plt.legend()

    plt.tight_layout()
    plt.show()

def check_symmetric(a:np.array):
    
    return ((a == a.T).all())# check for symmetry

def predict_mats(A, B, N):
    """
    Returns the MPC prediction matrices F and G for the system x^+ = A*x + B*u.
    
    Parameters:
    A : 2D array (n x n)
        State matrix.
    B : 2D array (n x m)
        Input matrix.
    N : int
        Prediction horizon length.
    
    Returns:
    F : 2D array
        Prediction matrix F.
    G : 2D array
        Prediction matrix G.
    """
    
    # Dimensions
    n = A.shape[0]
    m = B.shape[1]

    # Allocate matrices
    F = np.zeros((n * N, n))
    G = np.zeros((n * N, m * N))  # Ensure G has enough columns for the full horizon

    # Form row by row
    for i in range(N):
        # F matrix
        F[n * i:n * (i + 1), :] = np.linalg.matrix_power(A, i + 1)
        
        # G matrix
        for j in range(i + 1):
            G[n * i:n * (i + 1), m * j:m * (j + 1)] = np.linalg.matrix_power(A, i - j) @ B

    return F, G

def cost_mats(F, G, Q, R, P):
    from scipy.linalg import block_diag
    """
    Returns the MPC cost matrices H, L, and M for the system.
    
    Parameters:
    F : 2D array
        Prediction matrix F.
    G : 2D array
        Prediction matrix G.
    Q : 2D array
        State weighting matrix.
    R : 2D array
        Input weighting matrix.
    P : 2D array
        Terminal state weighting matrix.
    
    Returns:
    H : 2D array
        Hessian matrix.
    L : 2D array
        Linear term matrix.
    M : 2D array
        Constant term matrix.
    """
    
    # Get dimensions
    n = F.shape[1]
    N = F.shape[0] // n

    # Diagonalize Q and R
    Qd = np.kron(np.eye(N-1), Q) if N > 1 else np.array([])  # Handle case when N=1
    Qd = block_diag(Qd, P) if Qd.size > 0 else P
    Rd = np.kron(np.eye(N), R)

    # Hessian
    H = 2 * (G.T @ Qd @ G) + 2 * Rd

    # Linear term
    L = 2 * (G.T @ Qd @ F)

    # Constant term
    M = F.T @ Qd @ F + Q

    # Make sure the Hessian is symmetric
    H = (H + H.T) / 2

    return H, L, M


if __name__ == '__main__':
    main()


