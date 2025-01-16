using Pkg
Pkg.activate(".")
Pkg.add(url = "https://github.com/GenomicBreeding/GBCore.jl")
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
using Suppressor
using ProgressMeter

using DataFrames # tmp
