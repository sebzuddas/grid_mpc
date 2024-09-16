# Grid MPC

For the full problem definition, please see: [Assignment](/docs/ACS6116%20assignment.pdf)

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

```math

\Delta f(t) = 
\begin{pmatrix}
50 & 0 & 0 & 0 
\end{pmatrix}
\begin{pmatrix}
\Delta \omega(t)\\
\Delta p^m(t)\\
\Delta p^v(t)\\
\Delta p^{dr}(t)
\end{pmatrix}

```


### Explaining the Variables

1. $\Delta \omega (t)$
    1. Represents the deviation of **grid frequency** from the synchronous frequency of $50Hz$, using a normalised unit.
2. $\Delta p^m (t)$ 
    1. Represents the deviation from the **nominal mechanical** power output of the steam turbine, using a normalised unit.
3. $\Delta p^v (t)$ 
    1. Represents the deviation from the **governed mechanical** power output of the governor, using a normalised unit.
4. $\Delta p^{dr} (t)$
    1. Represents the **demand response**, using a normalised unit. This is the value used to reduce the difference in load, ie, $\Delta p^{load} (t)$, and the physical means for doing this is through the transfer of money (paying users).
5. $\Delta p^{load} (t)$
    1. Represents the deviation from **nominal net power demand**, using a normalised unit. This is the difference between the power _demanded by consumers_ and the _power generated by renewables_. 

### Tuning Parameters

There are a number of tuning parameters in the given problem, both within the controller and as part of the problem formulation. Within the controller, the $Q$ and $R$ matrices serve as ways for the controller to 'prioritise' different states or inputs as part of the optimisation procedure, and changing the prediction horizion $N$ changes how 'far ahead' the controller optimises for. Similarly, we have the ability to tune the demand response parameter $T^{dr}$ directly within the model. 



## Unconstrained

First attempt at the unconstrained controller

![unconstrained version 1](/docs/Figure_1_unconstrained.png)

Here we can see the unconstrained controller exerting control over the system. 
The controller itself can be validated as _stabilising_ through checking the spectral radius, which returns $SR = 0.8095604134222927\le 1$ hence the controller is stabilising.


#### Tuning Controller Variables

```math
Q = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&1
\end{pmatrix}
\cdot 
\begin{pmatrix}
\Delta \omega(t)\\
\Delta p^m(t)\\
\Delta p^v(t)\\
\Delta p^{dr}(t)
\end{pmatrix}
```


```math
R = \begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
\cdot 
\begin{pmatrix}
\Delta p^{m, ref}(t) \\
\Delta p^{dr, ref}(t) \\
\end{pmatrix}
```

---


## Constrained
### Attempt 1
the first attempt at tuning the constrained controller yields the following output

![constrained_1](docs/constrained_graphs_1.png)

This was achieved with matrices $Q$ and $R$

```math
Q = \begin{pmatrix}
0.0001&0&0&0\\
0&0.0001&0&0\\
0&0&0.00001&0\\
0&0&0&0.0000001
\end{pmatrix}
```


```math
R = \begin{pmatrix}
4&0\\
0&0.05
\end{pmatrix}
```
    

<!-- ### Modelling the Constraints
Constraints within MPC are more convenient if written as linear inequality constraints.  -->

<!--
We have the following constraints given as part of the problem. 


 ```math

\begin{bmatrix}



\end{bmatrix}
qwe
``` -->





## Disturbance Rejection



## Reference Tracking


