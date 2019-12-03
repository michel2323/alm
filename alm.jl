using Revise
using PowerModels
import InfrastructureModels
import Memento


import LinearAlgebra

# Run using Ipopt
# ipopt_solver = JuMP.with_optimizer(APM.Optimizer, tol=1e-6, print_level=5)
# pm = PowerModels.build_model("./data/matpower/case5.m", ACPPowerModel, PowerModels.post_opf) 
# solution = PowerModels.optimize_model!(pm, ipopt_solver; solution_builder = PowerModels.solution_opf!)
# @show solution["objective"]


module alm
using ReverseDiff
using LinearAlgebra
using Printf
using SparseArrays

export Params

mutable struct Params
    maxiter::Int64
    lambda::Union{Vector{Float64},Nothing}
    mu::Float64
    x::Union{Vector{Float64},Nothing}
    newton_epsilon::Float64
    al_epsilon::Float64 #
    status::Symbol

    # Callbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h::Function  # Can be nothing
    function Params(eval_f::Function, eval_g::Function, eval_grad_f::Function, 
     eval_jac_g::Function, eval_h::Function)
        new(0,nothing,0.0,nothing,0.0,0.0, :Empty, eval_f, eval_g, eval_grad_f,
            eval_jac_g, eval_h)
    end
end

function lagrangian(x::Vector{Float64},lambda::Vector{Float64},mu::Float64, param::Params)
    function psi(t, sigma, mu)
        if t - sigma/mu <= 0 
            return -sigma*t + t^2*mu/2
        else
            return - sigma^2/(2*mu)
        end
    end
    function grad_psi(t, sigma, mu)
        if t - sigma/mu <= 0 
            return -sigma + t*mu
        else
            return 0.0
        end
    end
    function f(x)
        ret = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3] # f(x)
        ret += -(sum(el^2 for el in x)-40)*lambda[1] + mu/2 * (sum(el^2 for el in x)-40)^2 # only equality constraint
        ret += psi(prod(el for el in x) - 25, lambda[2], mu) # psi as defined on p. 524 of Nocedal, Wright
        ret += psi(x[1] - 1, lambda[3], mu)
        ret += psi(x[2] - 1, lambda[4], mu)
        ret += psi(x[3] - 1, lambda[5], mu)
        ret += psi(x[4] - 1, lambda[6], mu)
        ret += psi(5 - x[1], lambda[7], mu)
        ret += psi(5 - x[2], lambda[8], mu)
        ret += psi(5 - x[3], lambda[9], mu)
        ret += psi(5 - x[4], lambda[10], mu)
    end

    function moif(x)
        ret = param.eval_f(x)
        g = zeros(Float64, 10)
        param.eval_g(x, g)
        ret += -g[1]*lambda[1] + mu/2 * g[1]^2 # only equality constraint
        ret += psi(g[2], lambda[2], mu) # psi as defined on p. 524 of Nocedal, Wright
        ret += psi(g[3], lambda[3], mu)
        ret += psi(g[4], lambda[4], mu)
        ret += psi(g[5], lambda[5], mu)
        ret += psi(g[6], lambda[6], mu)
        ret += psi(-g[7], lambda[7], mu) # <= requires switch of sign
        ret += psi(-g[8], lambda[8], mu)
        ret += psi(-g[9], lambda[9], mu)
        ret += psi(-g[10], lambda[10], mu)
    end

    function moig(x)
        grad_f = zeros(Float64, 4)
        param.eval_grad_f(x,grad_f)
        ret = grad_f
        rows = zeros(Float64, 10)
        cols = zeros(Float64, 4)
        njacobian = param.eval_jac_g(nothing, :Structure, nothing, nothing, nothing)
        rows = zeros(Float64, njacobian)
        cols = zeros(Float64, njacobian)
        values = zeros(Float64, njacobian)
        param.eval_jac_g(x, :Values, rows, cols, values)
        jac_g = sparse(rows, cols, values)
        g = zeros(Float64, 10)
        param.eval_g(x, g)

        ret += (-lambda[1] + mu*g[1]) * Array(jac_g[1,:])
        ret += grad_psi(g[2], lambda[2], mu) * Array(jac_g[2,:])
        ret += grad_psi(g[3], lambda[3], mu) * Array(jac_g[3,:])
        ret += grad_psi(g[4], lambda[4], mu) * Array(jac_g[4,:])
        ret += grad_psi(g[5], lambda[5], mu) * Array(jac_g[5,:])
        ret += grad_psi(g[6], lambda[6], mu) * Array(jac_g[6,:])
        ret += -grad_psi(-g[7], lambda[7], mu) * Array(jac_g[7,:])
        ret += -grad_psi(-g[8], lambda[8], mu) * Array(jac_g[8,:])
        ret += -grad_psi(-g[9], lambda[9], mu) * Array(jac_g[9,:])
        ret += -grad_psi(-g[10], lambda[10], mu) * Array(jac_g[10,:])
    end

    g = x -> ReverseDiff.gradient(f,x)
    h = x -> ReverseDiff.hessian(f,x)

    moif,g,h,moig
end

function normGradientLag(x::Vector{Float64}, lambda::Vector{Float64}, mu::Float64, param)
    f,g,h = lagrangian(x, lambda, mu, param)
    valueNG = norm(g(x))
end

function newton(x::Vector{Float64}, lambda::Vector{Float64}, mu::Float64, param::Params; verbose = true)
    f,g,h, g_ = lagrangian(x, lambda, mu, param)
    @show size(g(x))
    @show g(x)
    @show g_(x)
    res = 1e16
    iter = 0
    while norm(g(x)) > param.newton_epsilon
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

function solveProblem(param;verbose = false)
    # - Initialization
    maxiter = param.maxiter
    lambda = param.lambda
    mu = param.mu
    x = param.x
    k = 0
    # - Iterations
    while norm(lagrangian(x, lambda, mu, param)[2](x)) > param.al_epsilon && k<maxiter
        # - Newton 
        x = newton(x, lambda, mu, param, verbose=verbose)
        
        # - Updating dual variables and precision of the linesearch
        lambda = update_lambda(lambda, mu, x)
        if norm(violations(x)) < 1e-5
            mu = mu * 10.0
        end
        k = k+1
        
        # - Display of the results at each iteration
        if verbose == true
            println("")
            println("Iteration: ", k)
            println("mu: ", mu)
            println("lambda = ", lambda)
            println("x= ", x)
            println("Value of the Lagrangian Gradient norm: ", normGradientLag(x, lambda, mu, param))
            println("Value of the Constraint norm: ", violations(x))
            println("Hessian: ", lagrangian(x, lambda, mu, param)[3](x))
            println("objective: ", x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3] )
        end
    end
    if verbose == false
        println("")
        println("Total iterations: ", k)
        println("mu: ", mu)
        println("lambda = ", lambda)
        println("x= ", x)
        println("Value of the Lagrangian Gradient norm: ", normGradientLag(x, lambda, mu, param))
        println("Value of the Constraint norm: ", violations(x))
        println("Hessian: ", lagrangian(x, lambda, mu, param)[3](x))
        println("objective: ", x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3] )
    end
    param.status = :LOCALLY_SOLVED
    param.x = x
    x, lambda
end

function update_lambda(lambda::Vector{Float64}, mu::Float64, x::Vector{Float64})
# h is a vector containing the value of the different penalty functions for
# a given x
    h = Array{Float64,1}(undef, 10)
    # Equalities
    h[1] = lambda[1] - mu * (sum(el^2 for el in x) - 40)
    # Inequalities
    h[2] = update_lambda(lambda[2], mu, (prod(el for el in x) - 25))
    h[3] = update_lambda(lambda[3], mu, x[1] - 1)
    h[4] = update_lambda(lambda[4], mu, x[2] - 1)
    h[5] = update_lambda(lambda[5], mu, x[3] - 1)
    h[6] = update_lambda(lambda[6], mu, x[4] - 1)
    h[7] = update_lambda(lambda[7], mu, 5 - x[1])
    h[8] = update_lambda(lambda[8], mu, 5 - x[2])
    h[9] = update_lambda(lambda[9], mu, 5 - x[3])
    h[10] = update_lambda(lambda[10], mu, 5 - x[4])
    h
end

function update_lambda(lambda::Float64, mu::Float64, c::Float64)
    if -c + lambda/mu <= 0 
        return 0
    else
        return lambda - mu*c
    end
end

function ineq_violation(f)
    if f < 0.0
        return f
    else
        return 0.0
    end
end


function violations(x::Vector{Float64})
    c = Array{Float64}(undef, 10)
    c[1] = sum(el^2 for el in x) - 40
    # Inequalities
    c[2] = ineq_violation(prod(el for el in x) - 25)
    c[3] = ineq_violation(x[1] - 1)
    c[4] = ineq_violation(x[2] - 1)
    c[5] = ineq_violation(x[3] - 1)
    c[6] = ineq_violation(x[4] - 1)
    c[7] =  ineq_violation(5 - x[1])
    c[8] =  ineq_violation(5 - x[2])
    c[9] = ineq_violation( 5 - x[3])
    c[10] =  ineq_violation(5 - x[4])
    c
end

function createProblem(n, x_L, x_U,
    m, g_L, g_U,
    nele_jac, nele_hess,
    eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = nothing)

    param = Params(eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    param.maxiter = 1# maximum ALM iterations
    param.lambda = ones(Float64,10)
    param.mu = 1.0
    # param.x0 = [1.000, 4.743, 3.821, 1.379]
    param.x = [1.000, 5.0, 5.0, 1.0]
    param.al_epsilon = 1e-3
    param.newton_epsilon = 1e-7
    # param.model = model
    # - Solving the problem 
    x = param.x
    return param
end

include("MOI_wrapper.jl")
end

using .alm

using Ipopt

using JuMP
# - Augmented Lagrangian minimization
#  This file is an example of how to use the augmented Lagragngian algorithm
#  on a typical example
# - Parameters for test
# model = Model(with_optimizer(Ipopt.Optimizer))
model = Model(with_optimizer(alm.Optimizer))
@variable(model, x[1:4])
@NLconstraint(model, sum(x[i]^2 for i in 1:4) == 40.0)
@NLconstraint(model, prod(x[i] for i in 1:4) >= 25.0)
@NLconstraint(model, x[1] >= 1.0)
@NLconstraint(model, x[2] >= 1.0)
@NLconstraint(model, x[3] >= 1.0)
@NLconstraint(model, x[4] >= 1.0)
@NLconstraint(model, x[1] <= 5.0)
@NLconstraint(model, x[2] <= 5.0)
@NLconstraint(model, x[3] <= 5.0)
@NLconstraint(model, x[4] <= 5.0)
@NLobjective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
# @show model.inner

# @show model
# expr1 = objective_function_string(REPLMode, model)
# expr2 = objective_function_string(IJuliaMode, model)
# list = constraints_string(REPLMode, model)
# model.moi_backend.optimizer.model.inner.eval_f = expr1
optimize!(model)

# @show expr1
# @show typeof(expr1)
# @show expr2
# @show typeof(expr2)
# @show list
# @show typeof(list)

# @show model.nner
value.(x)
# param = alm.Params()
# param.maxiter = 1000# maximum ALM iterations
# param.lambda = ones(Float64,10)
# param.mu = 1.0
# # param.x0 = [1.000, 4.743, 3.821, 1.379]
# param.x0 = [1.000, 5.0, 5.0, 1.0]
# param.al_epsilon = 1e-3
# param.newton_epsilon = 1e-7
# param.model = model
# # - Solving the problem 
# x0 = param.x0
# println("lagrangian: ", alm.lagrangian(x0, zeros(Float64,10), 10000.0)[1](x0))
# println("objective: ", x0[1] * x0[4] * (x0[1] + x0[2] + x0[3]) + x0[3] )
# x, lambda = alm.solve(param; verbose = false)
