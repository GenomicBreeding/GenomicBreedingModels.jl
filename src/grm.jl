"""
    grmsimple(
        genomes::Genomes;
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        verbose::Bool = false
    )::Matrix{Float64}

Generate a simple genetic relationship matrix whose diagonals are likely inflated to allow inversion in dowstream analysis.

# Example
```jldoctest; setup = :(using GBCore, GBModels, LinearAlgebra)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> GRM = grmsimple(genomes);

julia> size(GRM), issymmetric(GRM)
((100, 100), true)

julia> det(GRM) > eps(Float64)
true
```
"""
function grmsimple(
    genomes::Genomes;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    verbose::Bool = false,
)::Matrix{Float64}
    # genomes = GBCore.simulategenomes(); idx_entries = nothing; idx_loci_alleles = nothing; verbose = true;
    # Check arguments while extracting the allele frequencies but first create a dummy phenomes struct
    phenomes_dummy = Phenomes(n = length(genomes.entries), t = 1)
    phenomes_dummy.entries = genomes.entries
    phenomes_dummy.populations = genomes.populations
    phenomes_dummy.traits = ["dummy_trait"]
    phenomes_dummy.phenotypes[:, 1] = rand(length(phenomes_dummy.entries))
    G, _, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes_dummy,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        add_intercept = false,
    )
    # Calculate a simple GRM
    GRM = G * G' ./ size(G, 2)
    # Inflate the diagonals is not invertible
    ϵ = 1e-4
    while det(GRM) < eps(Float64)
        GRM[diagind(GRM)] .+= ϵ
        ϵ .* 10
    end
    # Output
    if verbose
        UnicodePlots.heatmap(GRM)
    end
    GRM
end

"""
    grmploidyaware(
        genomes::Genomes;
        ploidy::Int64 = 2,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        verbose::Bool = false
    )::Matrix{Float64}

Generate a ploidy-aware genetic relationship matrix (see Bell et al (2017) and VanRaden et al (2008)) whose diagonals are likely inflated to allow inversion in dowstream analysis.

# Example
```jldoctest; setup = :(using GBCore, GBModels, LinearAlgebra)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> GRM_VanRaden = grmploidyaware(genomes);

julia> size(GRM_VanRaden), issymmetric(GRM_VanRaden)
((100, 100), true)

julia> det(GRM_VanRaden) > eps(Float64)
true
```
"""
function grmploidyaware(
    genomes::Genomes;
    ploidy::Int64 = 2,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    verbose::Bool = false,
)::Matrix{Float64}
    # genomes = GBCore.simulategenomes(); ploidy = 2; idx_entries = nothing; idx_loci_alleles = nothing; verbose = true;
    # Check arguments while extracting the allele frequencies but first create a dummy phenomes struct
    phenomes_dummy = Phenomes(n = length(genomes.entries), t = 1)
    phenomes_dummy.entries = genomes.entries
    phenomes_dummy.populations = genomes.populations
    phenomes_dummy.traits = ["dummy_trait"]
    phenomes_dummy.phenotypes[:, 1] = rand(length(phenomes_dummy.entries))
    G, _, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes_dummy,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        add_intercept = false,
    )
    # Calculate GRM via Bell et al (2017) and VanRaden et al (2008)
    q = mean(G, dims = 1)
    G_star = ploidy .* (G .- 0.5)
    q_star = ploidy .* (q .- 0.5)
    Z = G_star .- q
    GRM_VanRaden = (Z * Z') ./ (ploidy * sum(q .* (1 .- q)))
    # Inflate the diagonals is not invertible
    ϵ = 1e-4
    while det(GRM_VanRaden) < eps(Float64)
        GRM_VanRaden[diagind(GRM_VanRaden)] .+= ϵ
        ϵ .* 10
    end
    # Output
    if verbose
        UnicodePlots.heatmap(GRM_VanRaden)
    end
    GRM_VanRaden
end
