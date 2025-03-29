# GenomicBreedingModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GenomicBreeding.github.io/GenomicBreedingModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GenomicBreeding.github.io/GenomicBreedingModels.jl/dev/)
[![Build Status](https://github.com/GenomicBreeding/GenomicBreedingModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/GenomicBreeding/GenomicBreedingModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

Genomic prediction and population genetics models (including multi-generation breeding simulations) for GenomicBreeding.jl.

## Installation

This uses the [R package BGLR](https://github.com/gdlc/BGLR-R). We therefore need to install R and the BGLR package first. To help with this, Conda maybe used by loading the environment file: [`misc/GenomicBreeding_conda.yml`](misc/GenomicBreeding_conda.yml).

Then install GenomicBreedingModels.jl:

```julia
using Pkg
Pkg.add("https://github.com/GenomicBreeding/GenomicBreedingModels.jl")
```

If you wish to use neural network models, please install [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) on a GPU node. The install [LuxCUDA.jl](https://github.com/LuxDL/LuxCUDA.jl) for an NVIDIA GPU or [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) for a node with an AMD GPU:

Log into a GPU node, e.g.:

```shell
sinteractive --job-name="CUDA_install" --account="account_name" --partition="gpu" --gres=gpu:1
```

Install CUDA.jl:

```julia
using Pkg
Pkg.add("CUDA")
using CUDA
CUDA.set_runtime_version!(v"12.8") # modify to match you CUDA version: see shell> nvidia-smi
```

Then restart Julia to download the CUDA_runtime:

```julia
using CUDA
if CUDA.functional(true)
    Pkg.add("LuxCUDA")
    using LuxCUDA
else
    Pkg.add("AMDGPU")
    using AMDGPU
end
```

## Dev stuff:

### REPL prelude

```shell
julia --threads 3,1 --load test/interactive_prelude.jl
```

### Format and test

```shell
time julia --threads 3,1 test/cli_tester.jl
```
