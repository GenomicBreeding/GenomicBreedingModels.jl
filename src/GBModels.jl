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
using DataFrames

include("metrics.jl")
include("utils.jl")
include("bayes.jl")
include("linear.jl")
# include("non_linear.jl")
include("prediction.jl")
include("cross_validation.jl")

export metrics
export extractxyetc
export bglr, bayesian
export turing_bayesG, turing_bayesGs, turing_bayesGπ, turing_bayesGπs
export turing_bayesL, turing_bayesLs, turing_bayesLπ, turing_bayesLπs
export turing_bayesT, turing_bayesTπ
export turing_bayesG_logit
export ols, ridge, lasso, bayesa, bayesb, bayesc
export predict
export validate, cvmultithread!, cvbulk
export cvperpopulation, cvpairwisepopulation, cvleaveonepopulationout

end
