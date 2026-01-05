using Pkg
Pkg.activate(".")
# try
#     Pkg.update()
# catch
#     nothing
# end
using GenomicBreedingModels
using GenomicBreedingCore
using LinearAlgebra
using StatsBase
using Distributions
using Random
using Distances
using UnicodePlots
using GLMNet
using DataFrames
using Suppressor
using ProgressMeter
using DataFrames
using MixedModels
using MultivariateStats
using Optimization, Optimisers, Zygote
using Dates
# using Turing
# using BenchmarkTools, TuringBenchmarking
# using ReverseDiff
# using Lux, LuxCUDA
# using LuxCUDA, AMDGPU, oneAPI # GPU support (Metal is for MacOS)
