using Pkg
Pkg.activate(".")
try
    Pkg.add(url = "https://github.com/GenomicBreeding/GBCore.jl")
catch
    nothing
end
using GBModels
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
using DataFrames
using Suppressor
using ProgressMeter
using DataFrames
using MixedModels
