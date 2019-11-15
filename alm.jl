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
function lagrangian(x::Vector{Float64},lambda::Vector{Float64},c::Float64)
    # f is the Lagrangian: L(x) = t(x) + lambda'*penalty(x) +
    # 0.5*c||penalty(x)||^2
    # g is the gradient of the lagrangian
    # h is the hessian
    function f(x)
    ret = lambda[2]*(x[4]^2 + x[1] - 1.0) - (466523.0*log((x[1] + x[2] - 103.0/100.0)/((13.0*x[1])/100.0 + (13*x[2])/100.0 - 4.0/25.0)))/50.0 
    ret -= (225218.0*log((x[1] - 103.0/100.0)/(x[1] + (93.0*x[2])/100.0 - 103.0/100.0)))/25.0 
    ret -= lambda[6]*(- x[8]^2 + x[1] + x[2]) - (820437.0*log(-103.0/(100.0*((91.0*x[1])/100.0 - 103/100.0))))/100.0 
    ret += lambda[4]*(x[6]^2 + x[2] - 1.0) + (c*((x[4]^2 + x[1] - 1)^2 + (x[6]^2 + x[2] - 1)^2 + (- x[3]^2 + x[1])^2 + (- x[5]^2 + x[2])^2 + (x[7]^2 + x[1] + x[2] - 1)^2 + (- x[8]^2 + x[1] + x[2])^2))/2.0 
    ret += lambda[1]*(- x[3]^2 + x[1]) 
    ret += lambda[3]*(- x[5]^2 + x[2]) - lambda[5]*(x[7]^2 + x[1] + x[2] - 1.0)
    end

    g = x -> ReverseDiff.gradient(f,x)
    h = x -> ReverseDiff.hessian(f,x)

    f,g,h
end

function normGradientLag(func, x::Vector{Float64},lambda::Vector{Float64},c::Float64)
    # Evalue la fonction en xcurrent
    f,g,h = func( x, lambda, c)
    # calcule de la norme du gradient au point courant
    valueNG = norm(g(x))
end

function newton(func, x::Vector{Float64},lambda::Vector{Float64},c::Float64, epsilon)
    # Function which gives the steepest descent according to Newton's method
    # INPUT: fonction = name of the file containing the function of interest
    #        x = point at which we evaluate the 
    #        lambda = dual variable at which we evaluate the Lagrangian
    #        c = penalty value of the Lagrangian
    # OUTPUT: d = steepest direction according to Newton's method
    f,g,h = func(x, lambda, c)
    res = 1e16
    iter = 0
    while res > epsilon
        new_x = x - inv(h(x))*g(x)
        res = norm(new_x - x)
        x = new_x
        iter+=1
    end
    println("Newton iterations ", iter)
    x
end

function solve(param)
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
    x = param.x
    tau = param.tau
    alpha = param.alpha
    beta = param.beta
    epsilon2 = param.epsilon2
    epsilon = 1/c
    epsilon0 = epsilon
    eta0 = eta
    eta = eta/c^alpha
    fpen = pen(x)
    k = 1
    # - Iterations
    tStart = 0.0
    while (normGradientLag(lagrangian, x, lambda, 0.0) > epsilon2 || norm(fpen)^2 > epsilon2) && k<maxiter
        # - Linesearch
        x = newton(lagrangian, x, lambda, c, epsilon)
        fpen = pen(x)
        
        # - Updating dual variables and precision of the linesearch
        if (norm(fpen) <= eta)
            lambda = lambda + c*fpen
            epsilon = epsilon/c
            eta = eta/c^beta
        else
            c = tau*c
            epsilon = epsilon0/c
            eta = eta0/c^alpha
        end
        k = k+1
        
        # - Display of the results at each iteration
        println("")
        println("Iteration: ", k)
        println("Value of the Lagrangian Gradient norm: ", normGradientLag(lagrangian, x, lambda, 0.0))
        println("Value of the Constraint norm: ", norm(fpen))
    end
    println("x: ", x)
    println("x: ", lagrangian(x, zeros(Float64,6), 0.0)[1](x))
    x, lambda
end

function pen(x::Vector{Float64})
# h is a vector containing the value of the different penalty functions for
# a given x
    h = Array{Float64,1}(undef, 6)
    h[1] = x[1] -x[3]^2
    h[2] = x[1] +x[4]^2-1.0
    h[3] = x[2] -x[5]^2
    h[4] = x[2] +x[6]^2-1.0
    h[5] = 1.0-x[1]-x[2] -x[7]^2
    h[6] = 1.0-x[1]-x[2] +x[8]^2-1.0
    h
end

mutable struct Params
    maxiter::Int64
    lambda::Union{Vector{Float64},Nothing}
    c::Float64
    eta::Float64
    x::Union{Vector{Float64},Nothing}
    tau::Int64
    alpha::Float64
    beta::Float64
    epsilon::Float64
    epsilon2::Float64
    edit::Int64
    function Params()
        new(0,nothing,0,0.0,nothing,0,0.0,0.0,0.0,0.0,0)
    end
end
end

using .alm
# - Augmented Lagrangian minimization
#  This file is an example of how to use the augmented Lagragngian algorithm
#  on a typical example
# - Parameters for test
param = alm.Params()
param.maxiter = 2000 # maximum ALM iterations
param.lambda = Array([1., 1., 1., 1., 1., 1.])
param.c = 100.0
param.eta = 0.1 * 10.0^(0.1)
param.x = Array([0.7, 0.2, 0.1, sqrt(0.3), sqrt(0.2), sqrt(0.8), sqrt(0.1), sqrt(0.9)])
param.tau = 100
param.alpha = 0.1
param.beta = 0.9
param.epsilon2 = 0.5
param.edit = 2
# - Solving the problem 
x, lambda = alm.solve(param)
