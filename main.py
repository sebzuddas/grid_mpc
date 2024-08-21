import numpy as np
import do_mpc
from casadi import vertcat, MX, horzcat

def main():

    Ts = 0.1 # sampling time

    #Model params
    M = 8
    D = 0.8 
    R = 0.05 
    T_t = 0.5 
    T_g = 0.2 
    T_dr = 0.25 # demand response time (s)

    # limiting the amount of steam power that can be outputted

    #TODO: the inequality

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

    # simulate(100, x, u, A, B, E, C)

    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)


    # Define model parameters as CasADi symbols
    M = model.set_variable(var_type='parameter', var_name='M')
    D = model.set_variable(var_type='parameter', var_name='D')
    R = model.set_variable(var_type='parameter', var_name='R')
    T_t = model.set_variable(var_type='parameter', var_name='T_t')
    T_g = model.set_variable(var_type='parameter', var_name='T_g')
    T_dr = model.set_variable(var_type='parameter', var_name='T_dr')

    # defining state variables
    dw = model.set_variable(var_type = '_x', var_name = 'dw', shape=(1, 1))
    dpm = model.set_variable(var_type = '_x', var_name = 'dpm', shape=(1, 1))
    dpv = model.set_variable(var_type = '_x', var_name = 'dpv', shape=(1, 1))
    dpdr = model.set_variable(var_type = '_x', var_name = 'dpdr', shape=(1, 1))

    # defining input variables
    dpmref = model.set_variable(var_type='_u', var_name='dpmref', shape=(1, 1))
    dpdrref = model.set_variable(var_type='_u', var_name='dpdrref', shape=(1, 1))

    # Define the A, B, E matrices using the model parameters
    A = vertcat(
        horzcat(-D/M, 1/M, 0, 1/M),
        horzcat(0, -1/T_t, 1/T_t, 0),
        horzcat(-1/(R*T_g), 0, -1/T_g, 0),
        horzcat(0, 0, 0, -1/T_dr)
    )

    B = vertcat(
        horzcat(0, 0),
        horzcat(0, 0),
        horzcat(1/T_g, 0),
        horzcat(0, 1/T_dr)
    )

    E = vertcat(
        -1/M,
        0,
        0,
        0
    )

    # Define the ODEs based on the state-space equations
    x = vertcat(dw, dpm, dpv, dpdr)
    u = vertcat(dpmref, dpdrref)

    x_dot = A @ x + B @ u + E
    
    # Set the right-hand side (RHS) for each state
    model.set_rhs('dw', x_dot[0])
    model.set_rhs('dpm', x_dot[1])
    model.set_rhs('dpv', x_dot[2])
    model.set_rhs('dpdr', x_dot[3])

    model.setup()
    a = model.get_linear_system_matrices()
    
    print(a[0])

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=0.1)
    p_template = simulator.get_p_template()
    # type(p_template)
    # print(p_template.keys())
    simulator.set_p_fun(p_fun)

    simulator.setup()



def p_fun(t_now):
    p_template['M'] = 8
    p_template['D'] = 0.8
    p_template['R'] = 0.05
    p_template['T_t'] = 0.5
    p_template['T_g'] = 0.2
    p_template['T_dr'] = 0.25
    return p_template






if __name__ == "__main__":
    main()