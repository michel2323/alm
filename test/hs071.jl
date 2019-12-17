include("../src/alm.jl")

using .alm

using Ipopt

using JuMP

using Test

# model = Model(with_optimizer(Ipopt.Optimizer))
model = Model(with_optimizer(alm.Optimizer))
# @variable(model, x[1:5])
@variable(model, x[1:4])
# @NLconstraint(model, sum(x[i]^2 for i in 1:5) == 40.0)
# @NLconstraint(model, prod(x[i] for i in 1:5) >= 25.0)
@NLconstraint(model, sum(x[i]^2 for i in 1:4) == 40.0)
@NLconstraint(model, prod(x[i] for i in 1:4) >= 25.0)
@NLconstraint(model, x[1] >= 1.0)
@NLconstraint(model, x[2] >= 1.0)
@NLconstraint(model, x[3] >= 1.0)
@NLconstraint(model, x[4] >= 1.0)
# @NLconstraint(model, x[5] >= 1.0)
@NLconstraint(model, x[1] <= 5.0)
@NLconstraint(model, x[2] <= 5.0)
@NLconstraint(model, x[3] <= 5.0)
@NLconstraint(model, x[4] <= 5.0)
# @NLconstraint(model, x[5] <= 5.0)
# @NLobjective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3] + x[5]^2)
@NLobjective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
# set_start_value.(x, [1.0, 5.0, 5.0, 1.0, 1.0])
# set_start_value.(x, [1.0, 5.0, 3.26, 0.99, 1.53])
# set_start_value.(x, [4.42, 4.32, 1.34, 0.0, 0.0])
# set_start_value.(x, [1.0, 4.74, 3.82, 1.38])
set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
optimize!(model)
sol = value.(x)
@test sol[1] ≈ 1.0000000000000000 atol=1e-5
@test sol[2] ≈ 4.7429996418092970 atol=1e-5
@test sol[3] ≈ 3.8211499817883077 atol=1e-5
@test sol[4] ≈ 1.3794082897556983 atol=1e-5
# @test obj_val ≈ 17.014017145179164 atol=1e-5