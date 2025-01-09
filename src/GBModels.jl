module GBModels

using GBCore
using LinearAlgebra
using StatsBase
using Distributions
using Optimization
using Random
using Distances
using UnicodePlots, Plots
using GLMNet
using Turing
using Zygote
using Suppressor

include("metrics.jl")
include("bayes.jl")
include("linear.jl")

export metrics
export turing_bayesRR, turing_bayesLASSO, turing_bayesA, turing_bayesB
export ols, ridge, lasso, bayesRR

end
