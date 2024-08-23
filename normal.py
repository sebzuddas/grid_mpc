""" A version of main.py but with no libary, pure linear algebra"""

import numpy as np
import control as ct
import matplotlib.pyplot as plt


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


    A = np.array(
        [[-D/M, 1/M, 0, 1/M],
        [0, -1/T_t, 1/T_t, 0], 
        [-1/(R*T_g), 0, -1/T_g, 0], 
        [0, 0, 0, -1/T_dr]]) #System Matrix

    B = np.array(
        [[0, 0],
        [0, 0],
        [1/T_g, 0],
        [0, 1/T_dr]]) # Input Matrix

    E = np.array(
        [[-1/M],
        [0],
        [0],
        [0]]) # Disturbances
    
    C = np.array([[50], [0], [0], [0]]) # Output matrix


    # Weighting Matrices
    #TODO: Check
    # Q = C @ C.T # C'*C - weights on the states
    # R = np.transpose(B) @ B # B'*B

    # print(B.shape[1])
    m = B.shape[1]
    


    # in this case, it's better to have the weighting matrices as eye() since the other values result in sq matrices but with zero weighting. 
    Q = np.eye(4)
    R = np.eye(2)
    # print(Q)
    # print(R)
    
    
    K_inf = ct.dlqr(A, B, Q, R)
    K_2 = -1 * ct.place(A, B,[0.01, 0, 0, 0.01])
    # print(K_inf)
    # print(K_2)

    A_dlyap = (A+(B@K_2)).T
    Q_dlyap = (K_2.T @ R @ K_2) + Q
    
    # print(Q_dlyap)
    # print((Q_dlyap == Q_dlyap.T).all())# check for symmetry

    P = ct.dlyap(A_dlyap, Q_dlyap)
    # print(P)

    F, G = predict_mats(A, B, N) # where N is the prediction horizon

    H, L, M_cost = cost_mats(F, G, Q, R, P)

    # print(F, G, H, L, M_cost)

    S = np.linalg.solve(H, -L)
    # S becomes an m*N matrix (ie it is the set of optimal control inputs along the horizon, calculated in a receding manner)
    # print(S)
    
    KN = S[0:m, :]
    # print(KN)
    


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
        uopt = Uopt[0:m, :] # extract the first m from the sequence
        x = A @ x + B @ uopt

        k+=1

        xs.append(x)
        us.append(uopt)



    # Convert lists to numpy arrays for easier manipulation
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









def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

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


