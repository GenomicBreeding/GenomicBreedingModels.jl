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
export ols, ridge, lasso, bayesRR

end
