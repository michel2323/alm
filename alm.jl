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
    get_eq_con::Function
    get_ieq_le_con::Function
    get_ieq_ge_con::Function
    num_constraints::Int64
    function Params(eval_f::Function, eval_g::Function, eval_grad_f::Function, 
     eval_jac_g::Function, eval_h::Function, 
     get_eq_con::Function, get_ieq_le_con::Function, get_ieq_ge_con::Function)
        num_constraints = length(get_eq_con()) + length(get_ieq_le_con()) + length(get_ieq_ge_con()) 
        new(0,nothing,0.0,nothing,0.0,0.0, :Empty, eval_f, eval_g, eval_grad_f,
            eval_jac_g, eval_h, get_eq_con, get_ieq_le_con, get_ieq_ge_con, num_constraints)
    end
end

function trtofull(matrix)
    matrix = matrix + matrix'
    for i in 1:size(matrix,1)
        matrix[i,i] = matrix[i,i]/2
    end
    matrix
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
    function hess_psi(t, sigma, mu)
        if t - sigma/mu <= 0 
            return mu
        else
            return 0.0
        end
    end

    function moif(x)
        ret = param.eval_f(x)
        g = zeros(Float64, param.num_constraints)
        param.eval_g(x, g)
        for i in param.get_eq_con()
            ret += -g[i]*lambda[i] + mu/2 * g[i]^2 # only equality constraint
        end
        for i in param.get_ieq_ge_con()
            ret += psi(g[i], lambda[i], mu) # psi as defined on p. 524 of Nocedal, Wright
        end
        for i in param.get_ieq_le_con()
            ret += psi(-g[i], lambda[i], mu) # <= requires switch of sign
        end
        ret
    end

    function moig(x)
        grad_f = zeros(Float64, size(x))
        param.eval_grad_f(x,grad_f)
        ret = grad_f
        njacobian = param.eval_jac_g(nothing, :Structure, nothing, nothing, nothing)
        rows = zeros(Float64, njacobian)
        cols = zeros(Float64, njacobian)
        values = zeros(Float64, njacobian)
        param.eval_jac_g(x, :Values, rows, cols, values)
        jac_g = sparse(rows, cols, values)
        g = zeros(Float64, param.num_constraints)
        param.eval_g(x, g)

        for i in param.get_eq_con()
            ret += (-lambda[i] + mu*g[i]) * jac_g[i,:]
        end
        for i in param.get_ieq_ge_con()
            ret += grad_psi(g[i], lambda[i], mu) * jac_g[i,:]
        end
        for i in param.get_ieq_le_con()
            ret += -grad_psi(-g[i], lambda[i], mu) * jac_g[i,:]
        end
        ret
    end

    function moih(x)

        # Hessian
        nhessian = param.eval_h(nothing, :Structure, nothing, nothing, nothing, nothing, nothing)
        hrows = zeros(Float64, nhessian)
        hcols = zeros(Float64, nhessian)
        hvalues = zeros(Float64, nhessian)
        zero_lambda = zeros(Float64, param.num_constraints)
        param.eval_h(x, :Values, hrows, hcols, 1.0, zero_lambda, hvalues)
        hess_f = sparse(hrows, hcols, hvalues)

        # Jacobian
        njacobian = param.eval_jac_g(nothing, :Structure, nothing, nothing, nothing)
        jrows = zeros(Float64, njacobian)
        jcols = zeros(Float64, njacobian)
        jvalues = zeros(Float64, njacobian)
        param.eval_jac_g(x, :Values, jrows, jcols, jvalues)
        jac_g = sparse(jrows, jcols, jvalues)

        # Function
        g = zeros(Float64, param.num_constraints)
        param.eval_g(x, g)
        ret = 0
        # Equality constraints
        for i in param.get_eq_con()
            zero_lambda[i] = 1.0
            param.eval_h(x, :Values, hrows, hcols, 0.0, zero_lambda, hvalues)
            hes_g = sparse(hrows, hcols, hvalues)
            ret1 = (-lambda[i] + mu*g[i]) * trtofull(hes_g)
            ret2 = mu * jac_g[i,:] * jac_g[i,:]'
            ret = ret1 .+ ret2 .+ trtofull(hess_f)
            zero_lambda[i] = 0.0
        end

        # Inequality GE (>=) constraints
        for i in param.get_ieq_ge_con()
            zero_lambda[i] = 1.0
            param.eval_h(x, :Values, hrows, hcols, 0.0, zero_lambda, hvalues)
            hes_g = sparse(hrows, hcols, hvalues)
            ret1 = grad_psi(g[i], lambda[i], mu) * Array(trtofull(hes_g))
            ret2 = hess_psi(g[i], lambda[i], mu) * jac_g[i,:] * jac_g[i,:]' 
            ret = ret .+ ret1 .+ ret2
            zero_lambda[i] = 0.0
        end

        # Inequality LE (<=) constraints
        for i in param.get_ieq_le_con()
            zero_lambda[i] = 1.0
            param.eval_h(x, :Values, hrows, hcols, 0.0, zero_lambda, hvalues)
            hes_g = sparse(hrows, hcols, hvalues)
            ret1 = grad_psi(-g[i], lambda[i], mu) * Array(trtofull(hes_g))
            ret2 = hess_psi(-g[i], lambda[i], mu) * jac_g[i,:] * jac_g[i,:]' 
            ret = ret .+ ret1 .+ ret2
            zero_lambda[i] = 0.0
        end

        ret = Array(ret)
    end

    # g = x -> ReverseDiff.gradient(f,x)
    # h = x -> ReverseDiff.hessian(f,x)

    moif,moig,moih
end

function normGradientLag(x::Vector{Float64}, lambda::Vector{Float64}, mu::Float64, param)
    f,g,h = lagrangian(x, lambda, mu, param)
    valueNG = norm(g(x))
end

function newton(x::Vector{Float64}, lambda::Vector{Float64}, mu::Float64, param::Params; verbose = true)
    f,g,h = lagrangian(x, lambda, mu, param)
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
    constraints = similar(lambda)
    # - Iterations
    while norm(lagrangian(x, lambda, mu, param)[2](x)) > param.al_epsilon && k<maxiter
        # - Newton 
        x = newton(x, lambda, mu, param, verbose=verbose)
        
        # - Updating dual variables and precision of the linesearch
        param.eval_g(x, constraints)
        lambda = update_lambda!(lambda, mu, x, constraints, param)
        if norm(violations(x, constraints, param)) < 1e-5
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
            println("Value of the Constraint norm: ", violations(x, constraints, param))
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
        println("Value of the Constraint norm: ", violations(x, constraints, param))
        println("Hessian: ", lagrangian(x, lambda, mu, param)[3](x))
        println("objective: ", x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3] )
    end
    param.status = :LOCALLY_SOLVED
    param.x = x
    x, lambda
end

function update_lambda!(lambda::Vector{Float64}, mu::Float64, x::Vector{Float64}, g::Vector{Float64}, param::Params)
# h is a vector containing the value of the different penalty functions for
# a given x
    # Equalities
    for i in param.get_eq_con()
        lambda[i] = lambda[i] - mu * g[i]
    end

    # Inequalities
    for i in param.get_ieq_ge_con()
        lambda[i] = update_lambda(lambda[i], mu, g[i])
    end
    
    for i in param.get_ieq_le_con()
        lambda[i] = update_lambda(lambda[i], mu, -g[i])
    end
    lambda
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


function violations(x::Vector{Float64}, g::Vector{Float64}, param)
    c = Array{Float64}(undef, param.num_constraints)
    for i in param.get_eq_con()
        c[i] = g[i]
    end
    for i in param.get_ieq_ge_con()
        c[i] = ineq_violation(g[i])
    end
    for i in param.get_ieq_le_con()
        c[i] = ineq_violation(-g[i])
    end
    c
end

function createProblem(n, x_L, x_U,
    m, g_L, g_U,
    nele_jac, nele_hess,
    eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h,
    get_eq_con, get_ieq_le_con, get_ieq_ge_con)

    param = Params(eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h,
                   get_eq_con, get_ieq_le_con, get_ieq_ge_con)
    param.maxiter = 1000# maximum ALM iterations
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
optimize!(model)
value.(x)
