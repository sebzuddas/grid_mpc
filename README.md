# Grid MPC

The challenge is to design an MPC-based controller for maintaining the stability of an electricity grid that uses renewables for generation. 

The system block diagram is given as the following:

![System block diagram](docs/system_transfer_fcns.png)

The given state-space prediction model is:

```math

\frac{d}{dt}
\begin{pmatrix}
\Delta \omega(t)\\
\Delta p^m(t)\\
\Delta p^v(t)\\
\Delta p^{dr}(t)
\end{pmatrix}
= 
\begin{pmatrix}
-D/M & 1/M & 0 & 1/M\\
0 & -1/T^t & 1/T^t & 0 \\
-1/(R\cdot T^g) & 0 & -1/T^g &0 \\
0 & 0 & 0 & -1/T^{dr}
\end{pmatrix}
\begin{pmatrix}
\Delta \omega(t)\\
\Delta p^m(t)\\
\Delta p^v(t)\\
\Delta p^{dr}(t)
\end{pmatrix}
\\
+

\begin{pmatrix}
0 & 0\\
0 & 0\\
1/T^g & 0\\
0 & 1/T^{dr}\\
\end{pmatrix}

\begin{pmatrix}
\Delta p^{m, ref}(t) \\
\Delta p^{dr, ref}(t) \\
\end{pmatrix}

+
\begin{pmatrix}
-1/M\\
0\\
0\\
0\\
\end{pmatrix}
\Delta p^{load}(t)

```


## Unconstrained

First attempt at the unconstrained controller

![unconstrained version 1](/docs/Figure_1_attempt_1_unconstrained.png)

Here we can see the unconstrained controller successfully controlling the system. 
The controller itself can be validated as _stabilising_ through checking the spectral radius, which returns $SR = 0.8095604134222927\le 1$ hence the controller is stabilising. 

## Constrained



## Disturbance Rejection



## Reference Tracking


