import numpy as np
import do_mpc
from casadi import vertcat, MX, horzcat

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True



def main():

    Ts = 0.1 # sampling time

    ##### Defining the Model #####

    #Model params
    # M = 8
    # D = 0.8 
    # R = 0.05 
    # T_t = 0.5 
    # T_g = 0.2 
    # T_dr = 0.25 # demand response time (s)

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
    # print(a[0])

    ##### Configuring the Controller #####

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 5,
        't_step': Ts,
        'n_robust': 1,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)


    # quadratic objective functions for the mterm and the lterm of the objective function
    mterm = dw**2 + dpm**2 + dpv**2 + dpdr**2# not sure what meyer term is?
    lterm = dw**2 + dpm**2 + dpv**2 + dpdr**2#q matrix

    mpc.set_objective(mterm=mterm, lterm=lterm)

    # now set a penalty on control inputs
    mpc.set_rterm(
        dpmref = 1e-2,
        dpdrref = 1e-2
    )#rmatrix

    ## setting the state constraints ##
    mpc.bounds['lower', '_x', 'dw'] = 48
    mpc.bounds['upper', '_x', 'dw'] = 52
    
    # mpc.bounds['lower', '_x', 'dpm']
    # mpc.bounds['upper', '_x', 'dpm']

    # mpc.bounds['lower', '_x', 'dpv']
    # mpc.bounds['upper', '_x', 'dpv']

    # mpc.bounds['lower', '_x', 'dpdr']
    # mpc.bounds['upper', '_x', 'dpdr']

    ## setting input constraints ##

    mpc.bounds['lower', '_u', 'dpmref'] = -0.5
    mpc.bounds['upper', '_u', 'dpmref'] = 0.5

    mpc.bounds['lower', '_u', 'dpdrref'] = 0
    # mpc.bounds['upper', '_u', 'dpdrref']



    # implementing uncertainties on the parameters, if you don't want to include the uncertainties, then just put the numbers here
    mpc.set_uncertainty_values(
    M = np.array([8]),
    D = np.array([0.8]),
    R = np.array([0.05]),
    T_t = np.array([0.5]),
    T_g = np.array([0.2]),
    T_dr = np.array([0.25])
    )
    
    
    mpc.setup()


    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=Ts)
    
    # p_template is needed
    p_template = simulator.get_p_template()
    # type(p_template)
    # print(p_template.keys())

    def p_fun(t_now):
        p_template['M'] = 8
        p_template['D'] = 0.8
        p_template['R'] = 0.05
        p_template['T_t'] = 0.5
        p_template['T_g'] = 0.2
        p_template['T_dr'] = 0.25
        return p_template

    simulator.set_p_fun(p_fun)

    ##### Creating the control loop #####
    
    # it is possible to create a state estimator, but not always necessary

    # we want an initial state that isn't zero. 
    x0 = np.pi * np.array([1, 1, 1, 1]).reshape(-1, 1)
    simulator.x0 = x0
    mpc.x0 = x0

    
    simulator.setup()

    
    mpc.set_initial_guess() # used to set the initial guess of the optimisation problem

    

    mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
    sim_graphics = do_mpc.graphics.Graphics(simulator.data)

    fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
    fig.align_ylabels()

    for g in [sim_graphics, mpc_graphics]:
        # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
        g.add_line(var_type='_x', var_name='dw', axis=ax[0])
        g.add_line(var_type='_x', var_name='dpm', axis=ax[0])
        g.add_line(var_type='_x', var_name='dpv', axis=ax[0])
        g.add_line(var_type='_x', var_name='dpdr', axis=ax[0])

        # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
        g.add_line(var_type='_u', var_name='dpmref', axis=ax[1])
        g.add_line(var_type='_u', var_name='dpdrref', axis=ax[1])




    u0 = np.zeros((2,1))
    for i in range(200):
        simulator.make_step(u0)

    sim_graphics.plot_results()
    sim_graphics.reset_axes()




if __name__ == "__main__":
    main()