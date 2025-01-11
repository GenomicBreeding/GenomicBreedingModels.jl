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
using Suppressor

include("metrics.jl")
include("bayes.jl")
include("linear.jl")

export metrics
export turing_bayesG, turing_bayesGs, turing_bayesGπ, turing_bayesGπs
export turing_bayesL, turing_bayesLs, turing_bayesLπ, turing_bayesLπs
export turing_bayesT, turing_bayesTπ
export ols, ridge, lasso, bayesian

end
