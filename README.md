# GBModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GenomicBreeding.github.io/GBModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GenomicBreeding.github.io/GBModels.jl/dev/)
[![Build Status](https://github.com/GenomicBreeding/GBModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/GenomicBreeding/GBModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

Genomic prediction models for GenomicBreeding.jl.

## Installation

This uses the [R package BGLR](https://github.com/gdlc/BGLR-R). We therefore need to install R and the BGLR package first. To help with this, Conda maybe used by loading the environment file: [`misc/GenomicBreeding_conda.yml`](misc/GenomicBreeding_conda.yml).

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
time julia --threads 3,1 test/cli_tester.jl
```
