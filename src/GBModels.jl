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
using BenchmarkTools, TuringBenchmarking
using ReverseDiff
using RCall
using Suppressor
using ProgressMeter

using DataFrames # tmp

include("metrics.jl")
include("common.jl")
include("turing.jl")
include("bglr.jl")
include("linear.jl")
include("non_linear.jl")
include("cross_validation.jl")

export metrics
export extractxyetc, bayesian
export turing_bayesG, turing_bayesGs, turing_bayesGπ, turing_bayesGπs
export turing_bayesL, turing_bayesLs, turing_bayesLπ, turing_bayesLπs
export turing_bayesT, turing_bayesTπ
export turing_bayesG_logit
export bglr
export ols, ridge, lasso, bayesa, bayesb, bayesc
# export cvbulk

end
