""" A version of main.py but with no libary, pure linear algebra"""

import numpy as np
import control as ct
import control.matlab
import matplotlib.pyplot as plt
from scipy.linalg import kron

import math


def main():


    Ts = 0.1 # sampling time



    #### Defining the Model #####

    # Model params
    M = 8
    D = 0.8 
    R = 0.05 
    T_t = 0.5 
    T_g = 0.2 
    T_dr = 0.25 # demand response time (s)


    #### Continuous Time System Models ####

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
    
    Dc = np.zeros((p, m+1))
    # print(Dc)

    BcEc = np.concatenate((Bc, Ec), axis=1)
    print(BcEc)
    

    ##### Discrete-time model generation #####

    #obtains the discrete time prediction model
    sysc = ct.matlab.ss(Ac, BcEc, Cc, Dc)
    sysd = ct.matlab.c2d(sysc, Ts)
    A, BE, C, D = ct.matlab.ssdata(sysd)
    
    # Separate inputs B and disturbances E
    # print(BE)
    B = BE[:, 0:m]
    E = BE[:, m:]

    # print(A)
    # print(B)
    # print(C)
    # print(E)
    

    ### Weighting matrices ###
    # these are the tuning parameters
    
    # in this case, it's better to have the weighting matrices as eye() since the other values result in sq matrices but with zero weighting. 
    Q = np.eye(n) # squared state
    R = np.eye(m) # squared number of inputs
    
    N = 8 # prediction horizon
    
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
    # print(S)
    
    KN = S[0:m, :]
    
    # check for stability:
    spectral_radius(A, B, KN)




    ###### Constraints #########

    # state constraints?
    #xmax = [[rxy_max], [rxy_max], [500], [20]]

    # input constraints
    # umax = np.array([[0.5],
    #                  []])
   

    x0 = np.array([[1], 
                   [1], 
                   [0], 
                   [1]])# system starting states, dw(t), dpm(t), dpv(t), dpdr(t)
    

    u0 = np.array([[0], 
                   [0]]) # system inputs, dpm,ref(t) dpdr,ref(t)
    

    Uopt = S@x0#get the first set of optimal control inputs


    timesteps = 50
    k = 0
    x = x0

    xs = []
    us = []
    ys = []

    while k <= timesteps:

        Uopt = KN@x# calculate the optimal control sequence
        uopt = Uopt[0:n, :] # extract the first m from the sequence
        x = A @ x + B @ uopt
        y = C @ x
        k += 1
        
        xs.append(x)
        us.append(uopt)
        ys.append(y)


    plot_output(xs, us, ys)

    

def spectral_radius(A:np.array, B:np.array, KN:np.array):
     ### Calculating the stability of A+B@KN ###

    Z = A+B@KN
    eigenvals = np.linalg.eigvals(Z)
    sr = np.max(np.abs(eigenvals))
    
    print('Stabilising', sr) if sr < 1 else print('Not Stabilising', sr)

def plot_output(xs:list, us:list, ys:list):
    # # Convert lists to numpy arrays for easier manipulation
    # expect xs, us to be a list of numpy arrays
    xs = np.array(xs).squeeze()
    us = np.array(us).squeeze()
    ys = np.array(ys).squeeze()


    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Plotting states
    state_label = ['$\Delta \omega (t)$', '$\Delta p^m (t)$', '$\Delta p^v (t)$', '$\Delta p^{dr} (t)$']
    plt.subplot(3, 1, 1)
    for i in range(xs.shape[1]):
        plt.plot(xs[:, i], label=state_label[i])
    plt.title('State Trajectories')
    plt.xlabel('Time Step')
    plt.ylabel('State Values')
    plt.grid()
    plt.legend()

    # Plotting control inputs
    control_label = ['$\Delta p^{m, ref} (t)$', '$\Delta p^{dr, ref} (t)$']
    plt.subplot(3, 1, 2)
    for i in range(us.shape[1]):
        plt.plot(us[:, i], label=control_label[i])
    plt.title('Control Inputs')
    plt.xlabel('Time Step')
    plt.ylabel('Control Values')
    plt.grid()
    plt.legend()

    # Plotting outputs
    state_label = ['$\Delta \omega (t)$', '$\Delta p^m (t)$', '$\Delta p^v (t)$', '$\Delta p^{dr} (t)$']
    plt.subplot(3, 1, 3)
    plt.plot(ys, label='$\Delta \omega (t)$')
    # for i in range(ys.shape[1]):
    #     plt.plot(ys[:, i], label=state_label[i])
    plt.title('Outputs')
    plt.xlabel('Time Step')
    plt.ylabel('Output Values')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def check_symmetric(a:np.array):
    
    return ((a == a.T).all())# check for symmetry

def predict_mats(A:np.array, B:np.array, N:int):
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

def constraint_mats(F:np.array, G:np.array, Pu:np.array, qu:np.array, Px=None, qx=None, Pxf=None, qxf=None):
    """
    Returns the MPC constraints matrices for a system subject to constraints.

    The function computes the matrices Pc, qc, and Sc from:
    Pc * U(k) <= qc + Sc * x(k)

    Parameters:
        F, G : Prediction matrices
        Pu, qu : Input constraints
        Px, qx : State constraints (can be None if not applicable)
        Pxf, qxf : Terminal constraints (can be None if not applicable)

    Returns:
        Pc, qc, Sc : Matrices for constraints
    """

    # Input dimension
    m = Pu.shape[1]

    # State dimension
    n = F.shape[1]

    # Horizon length
    N = F.shape[0] // n

    # Number of input constraints
    ncu = qu.size

    # Number of state constraints
    ncx = qx.size if qx is not None else 0

    # Number of terminal constraints
    ncf = qxf.size if qxf is not None else 0

    # If state constraints exist, but terminal ones do not, then extend the
    # former to the latter
    if ncf == 0 and ncx > 0:
        Pxf = Px
        qxf = qx
        ncf = ncx

    ## Input constraints

    # Build "tilde" (stacked) matrices for constraints over horizon
    Pu_tilde = kron(np.eye(N), Pu)
    qu_tilde = np.kron(np.ones((N, 1)), qu)
    Scu = np.zeros((ncu * N, n))

    ## State constraints

    # Build "tilde" (stacked) matrices for constraints over horizon
    Px0_tilde = np.vstack([Px, np.zeros((ncx * (N - 1) + ncf, n))]) if ncx > 0 else np.zeros((ncf, n))
    
    if ncx > 0:
        Px_tilde = np.hstack([kron(np.eye(N - 1), Px), np.zeros((ncx * (N - 1), n))])
    else:
        Px_tilde = np.zeros((0, n * N))

    Pxf_tilde = np.hstack([np.zeros((ncf, n * (N - 1))), Pxf])
    Px_tilde = np.vstack([np.zeros((ncx, n * N)), Px_tilde, Pxf_tilde])
    qx_tilde = np.vstack([kron(np.ones((N, 1)), qx), qxf]) if ncx > 0 else qxf

    ## Final stack
    if Px_tilde.size == 0:
        Pc = Pu_tilde
        qc = qu_tilde
        Sc = Scu
    else:
        # Eliminate x for final form
        Pc = np.vstack([Pu_tilde, Px_tilde @ G])
        qc = np.vstack([qu_tilde, qx_tilde])
        Sc = np.vstack([Scu, -Px0_tilde - Px_tilde @ F])

    return Pc, qc, Sc
    
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

def check_ctrb_obsv(A, B, C=None):
    """Function to check controllability and observability for a given ss model"""

    n = A.shape[0]

    # controllability matrix
    Ct = ct.ctrb(A, B)
    rank_Ct = np.linalg.matrix_rank(Ct)
    controllable = (rank_Ct == n)
    
    # observability matrix
    if C is None:
        C = B

    Ob = ct.obsv(A, C)
    rank_Ob = np.linalg.matrix_rank(Ob)
    observable = (rank_Ob == n)


    return controllable, observable


if __name__ == '__main__':
    main()
