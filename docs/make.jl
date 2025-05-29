using GenomicBreedingModels
using Documenter

DocMeta.setdocmeta!(GenomicBreedingModels, :DocTestSetup, :(using GenomicBreedingModels); recursive = true)

makedocs(;
    modules = [GenomicBreedingModels],
    authors = "jeffersonparil@gmail.com",
    sitename = "GenomicBreedingModels.jl",
    format = Documenter.HTML(;
        canonical = "https://GenomicBreeding.github.io/GenomicBreedingModels.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/GenomicBreeding/GenomicBreedingModels.jl", devbranch = "main")
