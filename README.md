# GBModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GenomicBreeding.github.io/GBModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GenomicBreeding.github.io/GBModels.jl/dev/)
[![Build Status](https://github.com/GenomicBreeding/GBModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/GenomicBreeding/GBModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

Genomic prediction models for GenomicBreeding.jl.

## Dev stuff:

### REPL prelude

```shell
julia --threads 2,1 --load test/interactive_prelude.jl
```

### Format and test

```shell
time julia test/cli_tester.jl
```

### Docstring conventions

- Structs and main functions with title description, etc including Examples with doctests
- Methods, i.e. functions with the same names but different input types follow the usual Julia docstring pattern, i.e. the function signature, then some description, then details including parameter description, and finally examples with doctests