using Revise
using PowerModels
import InfrastructureModels
import Memento

import Ipopt

import JuMP


import LinearAlgebra

# Run using Ipopt
# ipopt_solver = JuMP.with_optimizer(Ipopt.Optimizer, tol=1e-6, print_level=5)
# pm = PowerModels.build_model("./data/matpower/case5.m", ACPPowerModel, PowerModels.post_opf) 
# solution = PowerModels.optimize_model!(pm, ipopt_solver; solution_builder = PowerModels.solution_opf!)
# @show solution["objective"]
module alm
using ReverseDiff
using LinearAlgebra
using Printf
function lagrangian(x::Vector{Float64},lambda::Vector{Float64},mu::Float64)
    # f is the Lagrangian: L(x) = t(x) + lambda'*penalty(x) +
    # 0.5*c||penalty(x)||^2
    # g is the gradient of the lagrangian
    # h is the hessian
    function psi(t, sigma, mu)
        if t - sigma/mu <= 0 
            return -sigma*t + t^2*mu/2
        else
            return - sigma^2/(2*mu)
        end
    end
    function f(x)
        ret = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3] # f(x)
        ret += -(sum(el^2 for el in x)-40)*lambda[1] + mu/2 * (sum(el^2 for el in x)-40)^2# only equality constraint
        ret += psi(prod(el for el in x) - 25, lambda[2], mu)
        ret += psi(x[1] - 1, lambda[3], mu)
        ret += psi(x[2] - 1, lambda[4], mu)
        ret += psi(x[3] - 1, lambda[5], mu)
        ret += psi(x[4] - 1, lambda[6], mu)
        ret += psi(5 - x[1], lambda[7], mu)
        ret += psi(5 - x[2], lambda[8], mu)
        ret += psi(5 - x[3], lambda[9], mu)
        ret += psi(5 - x[4], lambda[10], mu)
    end

    g = x -> ReverseDiff.gradient(f,x)
    h = x -> ReverseDiff.hessian(f,x)

    f,g,h
end

function normGradientLag(x::Vector{Float64},lambda::Vector{Float64},c::Float64)
    # Evalue la fonction en xcurrent
    f,g,h = lagrangian(x, lambda, c)
    # calcule de la norme du gradient au point courant
    valueNG = norm(g(x))
end

function newton(x::Vector{Float64},lambda::Vector{Float64},c::Float64, epsilon; verbose = true)
    # Function which gives the steepest descent according to Newton's method
    # INPUT: fonction = name of the file containing the function of interest
    #        x = point at which we evaluate the 
    #        lambda = dual variable at which we evaluate the Lagrangian
    #        c = penalty value of the Lagrangian
    # OUTPUT: d = steepest direction according to Newton's method
    f,g,h = lagrangian(x, lambda, c)
    res = 1e16
    iter = 0
    while norm(g(x)) > epsilon
        new_x = x - inv(h(x))*g(x)
        res = norm(new_x - x)
        x = new_x
        iter+=1
    end
    if verbose == true
        println("Newton iterations ", iter)
        println("Gradient ", norm(g(x)))
    end
    x
end

function solve(param;verbose = true)
    #  Function implementing the augmented Lagrangian algorithm of the
    #  Bierlaire's book.
    #  INPUTS: - fct: string name of the file containing Lagrangians (function,
    #  gradient, Hessian)
    #         - pen: string name of the file containing constraints (function,
    #  gradient, Hessian)
    #         - param: structure containing the following fields
    #                param.maxiter: max number of iterations
    #                param.lambda, param.c, param.x: initial variables used to calculate the
    #                Lagrangian
    #                param.eta, param.tau, param.alpha, param.beta: Augmented
    #                Lagrangian parameters (cf Bierlaire's book)
    #                param.epsilon2: stopping criterion of the algorithm
    #                param.linesearch: selected linesearch method
    #                param.edit: display option
    # OUTPUTS: - X: List containing the values of X along the iterations
    #          - lambda: List containg the lambda at optimum
    # - Initialization
    maxiter = param.maxiter
    lambda = param.lambda
    c = param.c
    eta = param.eta
    x = param.x0
    tau = param.tau
    alpha = param.alpha
    beta = param.beta
    epsilon2 = param.epsilon2
    epsilon = 1/c
    epsilon0 = epsilon
    eta0 = eta
    eta = eta/c^alpha
    fpen = pen(x)
    k = 0
    # - Iterations
    while (normGradientLag(x, lambda, c) > epsilon2 || norm(fpen)^2 > epsilon2) && k<maxiter
        # - Linesearch
        x = newton(x, lambda, c, epsilon, verbose=verbose)
        fpen = pen(x)
        
        # - Updating dual variables and precision of the linesearch
        # if (norm(fpen) <= eta)
            lambda = lambda - c*fpen
            # c = c*1.2
            # epsilon = epsilon/c
            # eta = eta/c^beta
        # else
            # epsilon = epsilon0/c
            # eta = eta0/c^alpha
        # end
        k = k+1
        
        # - Display of the results at each iteration
        if verbose == true
            println("")
            println("Iteration: ", k)
            println("c: ", c)
            println("epsilon: ", epsilon)
            println("pen = ", fpen)
            println("lambda = ", lambda)
            println("x= ", x)
            println("Value of the Lagrangian Gradient norm: ", normGradientLag(x, lambda, c))
            println("Value of the Constraint norm: ", norm(fpen))
            println("objective: ", x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3] )
        end
    end
    if verbose == false
        println("")
        println("Iteration: ", k)
        println("c: ", c)
        println("epsilon: ", epsilon)
        println("pen = ", fpen)
        println("lambda = ", lambda)
        println("x= ", x)
        println("Value of the Lagrangian Gradient norm: ", normGradientLag(x, lambda, c))
        println("Value of the Constraint norm: ", norm(fpen))
        println("objective: ", x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3] )
    end
    x, lambda
end

function pen(x::Vector{Float64})
# h is a vector containing the value of the different penalty functions for
# a given x
    h = Array{Float64,1}(undef, 10)
    h[1] = sum(el^2 for el in x) - 40
    h[2] = (prod(el for el in x) - 25)
    h[3] = x[1] - 1
    h[4] = x[2] - 1
    h[5] = x[3] - 1
    h[6] = x[4] - 1
    h[7] = 5 - x[1]
    h[8] = 5 - x[2]
    h[9] = 5 - x[3]
    h[10] = 5 - x[4]
    h
end

mutable struct Params
    maxiter::Int64
    lambda::Union{Vector{Float64},Nothing}
    c::Float64
    eta::Float64
    x0::Union{Vector{Float64},Nothing}
    tau::Int64
    alpha::Float64
    beta::Float64
    epsilon::Float64
    epsilon2::Float64
    function Params()
        new(0,nothing,0,0.0,nothing,0,0.0,0.0,0.0,0.0)
    end
end
end

using .alm
# - Augmented Lagrangian minimization
#  This file is an example of how to use the augmented Lagragngian algorithm
#  on a typical example
# - Parameters for test
param = alm.Params()
param.maxiter = 100# maximum ALM iterations
param.lambda = ones(Float64,10)
param.c = 1000
param.eta = 0.1 * 10.0^(0.1)
param.x0 = [1.000, 4.743, 3.821, 1.379]
param.tau = 100
param.alpha = 0.1
param.beta = 0.9
param.epsilon2 = 0.5
# - Solving the problem 
x0 = param.x0
println("lagrangian: ", alm.lagrangian(x0, zeros(Float64,10), 10000.0)[1](x0))
println("objective: ", x0[1] * x0[4] * (x0[1] + x0[2] + x0[3]) + x0[3] )
x, lambda = alm.solve(param; verbose = true)
