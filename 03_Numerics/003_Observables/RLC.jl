# Flat modeling of an RLC circuit
# TOpology similar to Michael Tiller JSML demo: https://shorturl.at/LQIFT
using ModelingToolkit, DifferentialEquations, Plots

# Define the independent variable time
@independent_variables t

# Define our variables: variable(t) = initial condition
@variables ΔV(t)=0 v_L(t)=0 i_L(t)=0 v_R(t)=0 i_R(t)=0 v_C(t)=0 i_C(t)=0

# Define our parameters
@parameters R=100 L=1 C=0.001

# Define our differential: takes the derivative with respect to `t`
D = Differential(t)

# Define the set of algebraic differential equations
eqs = [ΔV  ~ ifelse(t<=1, 0, 32)
       i_L ~ i_R + i_C
       ΔV - v_L ~ v_R
       v_R ~ v_C
       v_R ~ i_R * R
       v_L ~ L * D(i_L)
       i_C ~ C * D(v_C)]

# Bring these pieces together into an ODESystem with independent variable t
@mtkbuild sys = ODESystem(eqs, t)

# Convert from a symbolic to a numerical problem to simulate
tspan = (0.5, 2.5)
prob = ODEProblem(sys, [], tspan)

# Solve the ODE
sol = solve(prob)

# Plot the solution
p1 = plot(sol, idxs = [i_L, i_R, i_C], title = "Currents")
p2 = plot(sol, idxs = [ΔV, v_L, v_R, v_C], title = "Voltages")

plot(p1, p2, layout = (2, 1))
