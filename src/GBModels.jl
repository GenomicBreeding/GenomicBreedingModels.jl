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
using DataFrames
using Suppressor
using ProgressMeter
using DataFrames
using Optimization, Optim
using MixedModels
using MultivariateStats
using Dates
using Lux, Optimisers, Zygote
# using LuxCUDA, AMDGPU, oneAPI # GPU support (Metal is for MacOS)

include("metrics.jl")
include("prediction.jl")
include("transformation.jl")
include("grm.jl")
include("gwas.jl")
include("bayes.jl")
include("dl.jl")
include("linear.jl")
include("non_linear.jl")
include("cross_validation.jl")

export metrics
export extractxyetc, predict
export grmsimple, grmploidyaware
export gwasprep, gwasols, gwaslmm, loglikreml, gwasreml
export square, invoneplus, log10epsdivlog10eps, mult, addnorm, raise
export transform1, transform2, epistasisfeatures, @string2operations, reconstitutefeatures
export bglr, bayesian
export turing_bayesG, turing_bayesGs, turing_bayesGπ, turing_bayesGπs
export turing_bayesL, turing_bayesLs, turing_bayesLπ, turing_bayesLπs
export turing_bayesT, turing_bayesTπ
export turing_bayesG_logit
export mlp
export ols, ridge, lasso, bayesa, bayesb, bayesc
export validate, cvmultithread!, cvbulk
export cvperpopulation, cvpairwisepopulation, cvleaveonepopulationout

end
