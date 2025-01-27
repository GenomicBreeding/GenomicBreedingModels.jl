# GBModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GenomicBreeding.github.io/GBModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GenomicBreeding.github.io/GBModels.jl/dev/)
[![Build Status](https://github.com/GenomicBreeding/GBModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/GenomicBreeding/GBModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

Genomic prediction models for GenomicBreeding.jl.

## Installation

This uses the [R package BGLR](https://github.com/gdlc/BGLR-R) called [via RCall.jl](https://github.com/JuliaInterop/RCall.jl). We therefore need to install R and the BGLR package first:

```julia
using Pkg
R_HOME='*'
Pkg.build("RCall")
```

Then install GBModels.jl:

```julia
using Pkg
Pkg.add("https://github.com/GenomicBreeding/GBModels.jl")
```

## Dev stuff:

### REPL prelude

```shell
julia --threads 3,1 --load test/interactive_prelude.jl
```

### Format and test

```shell
time julia test/cli_tester.jl
```

### Docstring conventions

- Structs and main functions with title description, etc including Examples with doctests
- Methods, i.e. functions with the same names but different input types follow the usual Julia docstring pattern, i.e. the function signature, then some description, then details including parameter description, and finally examples with doctests