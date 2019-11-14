using PowerModels
#import InfrastructureModels
#import Memento
# Memento.setlevel!(Memento.getlogger(PowerModels), "error")

import Ipopt
# import SCS
# import Pavito
# import Juniper

import JuMP
#import JSON

#import LinearAlgebra

# Run using Ipopt
ipopt_solver = JuMP.with_optimizer(Ipopt.Optimizer, tol=1e-6, print_level=5)
pm = PowerModels.build_model("./data/matpower/case5.m", ACPPowerModel, PowerModels.post_opf) 
solution = PowerModels.optimize_model!(pm, ipopt_solver; solution_builder = PowerModels.get_solution)
@show solution["objective"]


