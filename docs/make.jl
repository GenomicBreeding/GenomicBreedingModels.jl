using GBModels
using Documenter

DocMeta.setdocmeta!(GBModels, :DocTestSetup, :(using GBModels); recursive=true)

makedocs(;
    modules=[GBModels],
    authors="jeffersonparil@gmail.com",
    sitename="GBModels.jl",
    format=Documenter.HTML(;
        canonical="https://GenomicBreeding.github.io/GBModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/GenomicBreeding/GBModels.jl",
    devbranch="main",
)
