using Pkg
Pkg.activate(".")
try
    Pkg.update()
catch
    nothing
end
using GenomicBreedingModels
using GenomicBreedingCore
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
