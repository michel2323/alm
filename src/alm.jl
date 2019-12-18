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
        ret = copy(grad_f)
        njacobian = param.eval_jac_g(nothing, :Structure, nothing, nothing, nothing)
        rows = zeros(Float64, njacobian)
        cols = zeros(Float64, njacobian)
        values = zeros(Float64, njacobian)
        param.eval_jac_g(x, :Values, rows, cols, values)
        jac_g = sparse(rows, cols, values)
        g = zeros(Float64, param.num_constraints)
        param.eval_g(x, g)
        ret1 = zeros(Float64, size(x))
        ret2 = zeros(Float64, size(x))
        ret3 = zeros(Float64, size(x))
        for i in param.get_eq_con()
            ret += (-lambda[i] + mu*g[i]) * jac_g[i,:]
            ret1 += (-lambda[i] + mu*g[i]) * jac_g[i,:]
        end
        for i in param.get_ieq_ge_con()
            ret += grad_psi(g[i], lambda[i], mu) * jac_g[i,:]
            ret2 += grad_psi(g[i], lambda[i], mu) * jac_g[i,:]
        end
        for i in param.get_ieq_le_con()
            ret += -grad_psi(-g[i], lambda[i], mu) * jac_g[i,:]
            ret3 += -grad_psi(-g[i], lambda[i], mu) * jac_g[i,:]
        end
        # @show norm(grad_f)
        # @show norm(ret1)
        # @show norm(ret2)
        # @show norm(ret3)
        ret4 = grad_f + ret1 + ret2 + ret3
        # @show norm(ret4)
        # @show norm(ret)
        ret4
    end

    function moih(x)

        # Hessian
        nhessian = param.eval_h(nothing, :Structure, nothing, nothing, nothing, nothing, nothing)
        # @show nhessian
        hrows = zeros(Float64, nhessian)
        hcols = zeros(Float64, nhessian)
        hvalues = zeros(Float64, nhessian)
        zero_lambda = zeros(Float64, param.num_constraints)
        param.eval_h(x, :Values, hrows, hcols, 1.0, zero_lambda, hvalues)
        # @show hrows
        # @show hcols
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
        ret = zeros(Float64, size(hess_f,1), size(hess_f, 2))
        # Equality constraints
        for i in param.get_eq_con()
            zero_lambda[i] = 1.0
            param.eval_h(x, :Values, hrows, hcols, 0.0, zero_lambda, hvalues)
            hes_g = sparse(hrows, hcols, hvalues)
            ret1 = (-lambda[i] + mu*g[i]) * trtofull(hes_g)
            ret2 = mu * jac_g[i,:] * jac_g[i,:]'
            # @show size(ret)
            # @show size(ret1)
            # @show size(ret2)
            # @show size(trtofull(hess_f))
            ret = ret + ret1 + ret2 + trtofull(hess_f)
            zero_lambda[i] = 0.0
        end
        # @show "here2"

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
        # @show "here3"

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

function wolfe(xold, f, g, d, gamma);
    beta = 10^-4
    lambda = 50
    alpha_r = Inf
    alpha_l = 0.0
    alpha = 1.0
    k=0
    maxit = 200
    f_old = f(xold)
    g_old = g(xold)
    while k<maxit 
        f_new = f(xold + alpha*d)
        g_new = g(xold + alpha*d)
        wolfe_1 = f_new > f_old + alpha*beta*g_old'*d
        wolfe_2 = g_new'*d < gamma*g_old'*d
        if wolfe_1 == 0 && wolfe_2==0
            return alpha
        end
        # Check 1st Wolfe condition
        if wolfe_1 == true 
            alpha_r = alpha
            alpha = (alpha_l + alpha_r)/2;
        end
        # Check 2nd Wolfe condition
        if wolfe_2 == true && wolfe_1== false
            alpha_l = alpha
            if alpha_r==Inf
                alpha = lambda*alpha
            else
                alpha = (alpha_l + alpha_r)/2;
            end
        end
        k = k+1;
    end
    alpha
end

function newton(x::Vector{Float64}, lambda::Vector{Float64}, mu::Float64, param::Params; verbose = true)
    f,g,h = lagrangian(x, lambda, mu, param)
    res = 1e16
    iter = 0
    alpha = 1.0
    failed = false
    while norm(g(x)) > param.newton_epsilon && iter < 25
        hess = h(x)
        failed = false
        try
            global invh = inv(hess)
        catch e
            try
                global invh = inv(hess + (norm(g(x)) * sparse(I,size(hess,1), size(hess,2))))
            catch e
                @show norm(g(x))
                @show hess
                error("Singular after regularization")
                failed = true
            end
        end
        if any(isnan.(invh)) && failed == false
            global invh = inv(hess + (norm(g(x)) * sparse(I,size(hess,1), size(hess,2))))
            if any(isnan.(invh))
                @show norm(g(x))
                @show hess
                error("NaNs in inverted Hessian")
                failed = true
            end
        end
        if !failed
            d = invh*g(x)
            alpha  = wolfe(x, f, g, -d, 0.99);
        else
            # Do steepest descent
            d = Array(g(x)*f(x))
            alpha  = wolfe(x, f, g, -d, 0.1);
        end
        new_x = x - alpha*d
        res = norm(new_x - x)
        x = new_x
        iter+=1
    end
    if verbose == true
        println("Newton iterations ", iter)
        println("Gradient ", norm(g(x)))
        if failed == true
            println("Used steepest descent")
        end
    end
    x
end

function solveProblem(param;verbose = true)
    # - Initialization
    maxiter = param.maxiter
    lambda = param.lambda
    mu = param.mu
    @show param.x
    x = param.x
    k = 0
    constraints = similar(lambda)
    eta = 1e-5
    # - Iterations
    while (norm(lagrangian(x, lambda, mu, param)[2](x)) > param.al_epsilon || norm(violations(x, constraints, param)) > 1e-5) && k<maxiter
        # - Newton 
        x = newton(x, lambda, mu, param, verbose=verbose)
        
        # - Updating dual variables and precision of the linesearch
        param.eval_g(x, constraints)
        lambda = update_lambda!(lambda, mu, x, constraints, param)
        if norm(violations(x, constraints, param)) < eta
            if mu < 1000
                mu = mu * 10.0
            end
            if eta > 1e-5
                eta = eta / 10.0
            end
        else
            eta = eta * 10.0
        end
        k = k+1
        
        # - Display of the results at each iteration
        if verbose == true
            println("")
            println("Iteration: ", k)
            println("mu: ", mu)
            println("eta: ", eta)
            # println("lambda = ", lambda)
            # println("x= ", x)
            println("Norm of Lagrangian gradient: ", normGradientLag(x, lambda, mu, param))
            println("Norm of constraint violations: ", norm(violations(x, constraints, param)))
            # println("Hessian: ", lagrangian(x, lambda, mu, param)[3](x))
            println("objective: ", param.eval_f(x))
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
        # println("Hessian: ", lagrangian(x, lambda, mu, param)[3](x))
        println("objective: ", param.eval_f(x) )
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
    param.maxiter = 10000 # maximum ALM iterations
    param.lambda = ones(Float64,param.num_constraints)
    param.mu = 1.0
    param.al_epsilon = 1e-4
    param.newton_epsilon = 1e-8
    return param
end

include("MOI_wrapper.jl")
end
