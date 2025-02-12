# Flat modeling of suspended mass
# The aim of this model is to demonstrate the calibration of parameters based on data
# 1. First, the suspended mass is modeled and simulated.
# 2. Then, the data are modified with noise.
# 3. Finally, the parameters are calibrated based on the noisy data  starting from a different initial condition.
# 4. The calibrated parameters are used to simulate the fitted data and compare the results.
# 5. Results are analyzed and discussed.

using ModelingToolkit, DifferentialEquations, Plots
using ModelingToolkit: t_nounits as t, D_nounits as D

# 1. Baseline simulation

# Define our parameters
# The mass `m` is defined in [kg], the spring stiffness `k` in [N/m], and the viscous friction coefficient of the damper `c` is defined in [N.s/m]
# The acceleration due to gravity `g` is defined in [m/s²]
@parameters m k c g xₘ₀

# Define our variables: variable(t) = initial condition
@variables  xₘ(t) ẋₘ(t) ẍₘ(t) Fᵢ(t) Fₘ(t) Fₖ(t) F_c(t)

# Define the set of algebraic differential equations
eqs = [Fᵢ  ~ + Fₖ - F_c - Fₘ
       Fᵢ  ~ m*ẍₘ
       Fₘ  ~ m*g
       F_c ~ c*ẋₘ
       Fₖ  ~ k*(xₘ₀-xₘ)
       ẋₘ  ~ D(xₘ)
       ẍₘ  ~ D(ẋₘ)]

# Bring these pieces together into an ODESystem with independent variable t
@mtkbuild odesys = ODESystem(eqs, t)

# Convert from a symbolic to a numerical problem to simulate
tstart = 0
tend = 5
tspan = (tstart, tend)
steps = 0.02
timesteps = tstart:steps:tend
odeprob = ODEProblem(odesys, [xₘ => xₘ₀, ẋₘ => 0], tspan, [xₘ₀ => 1, g => 9.81, k => 1000, m => 10, c => 30])

"""
```julia
param_names_values(sys::AbstractODESystem) where {iip}
```
Convenience function to extract the names and values of the parameters of a problem.
Returns a dictionary with the names of the parameters as keys and their values as values.
"""
function param_names_values(prob::ODEProblem)
    return Dict(parameters(prob.f.sys) .=> parameter_values(prob)[1])
end

# Solve the ODE
sol = solve(odeprob, Tsit5(); saveat = timesteps)

# Plot the solution
p1 = plot(sol, idxs = [xₘ, ẋₘ, ẍₘ], title = "Dynamics");
p2 = plot(sol, idxs = [Fᵢ, Fₘ, Fₖ, F_c], title = "Forces");

plot(p1, p2, layout = (2, 1))

# 2. Generate noisy data
# extracting 1 every 0.2secs of the solution for variables t and xₘ
xₘ_data = Array(sol)

# Plot xₘ_data against t_data
plot(sol, idxs = [xₘ], title = "Position")
scatter!(sol.t, xₘ_data[1, :], xlabel="Time (s)", ylabel="Position (m)", title="Position vs Time")
# add some random noise
xₘ_data = xₘ_data + 0.005 * randn(size(xₘ_data))
#Plot noisy xₘ_data against t_data
scatter!(sol.t, xₘ_data[1, :], xlabel="Time (s)", ylabel="Position (m)", title="Position vs Time")

# 3. Calibrate parameters based on noisy data
using SymbolicIndexingInterface: parameter_values, state_values
using SciMLStructures: Tunable, replace, replace!

# Define a loss function to be minimized during optimization
function loss(x, p)
    odeprob = p[1] # ODEProblem stored as parameters to avoid using global variables
    ps = parameter_values(odeprob) # obtain the parameter object from the problem
    ps = replace(Tunable(), ps, x) # create a copy with the values passed to the loss function
    # remake the problem, passing in our new parameter object
    newprob = remake(odeprob; p = ps)
    timesteps = p[2]
    sol = solve(newprob, Tsit5(); saveat = timesteps)
    data = Array(sol)
    truth = p[3]
    return sum((truth .- data) .^ 2) / length(truth)
end

# Define a callback function to monitor optimization progress
function callback(state, l)
    display(l)
    display(state)
    ps = parameter_values(odeprob) # obtain the parameter object from the problem
    ps = replace(Tunable(), ps, state.u) # create a copy with the values passed to the loss function
    newprob = remake(odeprob; p = ps)
    sol = solve(newprob, saveat = timesteps)
    plt = plot(sol.t, Array(sol)[1,:], label = "Parameter fitting")
    scatter!(plt, sol.t, xₘ_data[1,:], label = "Noisy data")
    display(plt)
    return false
end

using Optimization
using OptimizationOptimJL

# manually create an OptimizationFunction to ensure usage of `ForwardDiff`, which will
# require changing the types of parameters from `Float64` to `ForwardDiff.Dual`
optfn = OptimizationFunction(loss, Optimization.AutoForwardDiff())
# parameter object is a tuple, to store differently typed objects together
# parameters(odesys) returns [xₘ₀, g, k, m, c]
# Solution is [xₘ₀ => 1, g => 9.81, k => 1000, m => 10, c => 30]
p_guess = [1, 9.81, 100.0, 30.0, 1.0]
p_min = [0, 9.80, 100.0, 1, 0.5]
p_max = [2, 9.82, 5000.0, 100, 500.0]
optprob = OptimizationProblem(
    optfn, p_guess, (odeprob, timesteps, xₘ_data), lb = p_min, ub = p_max)

sol_optim = solve(
    optprob, 
    BFGS(),
    callback = callback)

sol_optim.u
# 4. Simulate the fitted data to analyze the calibration
odeprob_fitting = ODEProblem(
    odesys, 
    [xₘ => xₘ₀, ẋₘ => 0], 
    tspan, 
    sol_optim.u)

sol_fitting = solve(odeprob_fitting, Tsit5(); saveat = timesteps)

# Plot the fitted, noisy and ideal data
scatter(sol.t, xₘ_data[1, :], xlabel="Time (s)", ylabel="Position (m)", title="Position vs Time", label = "Noisy data")
plot!(sol_fitting, idxs = [xₘ], label="Fitted data")
plot!(sol, idxs = [xₘ], label = "Ideal data")

# 5. Results analyzes and discussion

p1_optim = plot(sol_fitting, idxs = [xₘ, ẋₘ, ẍₘ], title = "Optim Dynamics");
p2_optim = plot(sol_fitting, idxs = [Fᵢ, Fₘ, Fₖ, F_c], title = "Optim Forces");
plot(p1, p2, p1_optim, p2_optim, layout = (2, 2))

# The model appears as successfully calibrated based on the noisy data, and the fitted data closely resembles the ideal data.

# However, let's analyze the results in more detail to ensure the calibration is accurate.
param_names_values(odeprob) == param_names_values(odeprob_fitting)

# The parameter values from the initial problem and from the fitting do not match.
# This is however not surprising as exact matching is not expected due to the noise in the data (and there might always be some tolerances in play).

# Let's print the parameter values for both problems to compare them.
initial_param = param_names_values(odeprob) 
fitting_param = param_names_values(odeprob_fitting)

# These are quite different!
# Why is so?
# The answer lies in the physics of our system - not in the mathematical process of the parameter fitting.
# Our model is a simple damped harmonic oscillator. The parameters are related to each other in a specific way.
# The damped natural frequency is given by w = sqrt(w₀^2 - zeta^2) - with w₀ = sqrt(k/m) and the damping ratio is zeta = c/(2*sqrt(k*m)).
# So, while we tried to optimize the parameters independently, there are actually only two dimensions of freedom in our system.
# Let's look at how these variables compare for the initial and fitted parameters.

"""
harmonic_oscillator_parameters(param::Dict) -> Tuple
"""
function harmonic_oscillator_parameters(param::Dict)
    w₀ = sqrt(param[k]/param[m])
    zeta = param[c]/(2*sqrt(param[k]*param[m]))
    w = sqrt(w₀^2 - zeta^2)
    return w, w₀, zeta
end

harmonic_oscillator_parameters(initial_param)
harmonic_oscillator_parameters(fitting_param)

# The damped natural frequency w is the same for both the initial and fitted parameters - within some small tolerances.