"""
    grmsimple(
        genomes::Genomes;
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        verbose::Bool = false
    )::Matrix{Float64}

Generate a simple genetic relationship matrix (GRM) from genomic data.

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to select specific entries/individuals
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to select specific loci/alleles
- `verbose::Bool`: If true, displays a heatmap visualization of the GRM

# Returns
- `Matrix{Float64}`: A symmetric positive definite genetic relationship matrix

# Details
The function computes a genetic relationship matrix by:
1. Converting genomic data to a numerical matrix
2. Computing GRM as G * G' / ncol(G)
3. Adding small positive values to diagonal elements if necessary to ensure matrix invertibility

# Notes
- The resulting matrix is always symmetric
- Diagonal elements may be slightly inflated to ensure matrix invertibility
- The matrix dimensions will be n×n where n is the number of entries/individuals

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
        ϵ *= 10
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

Generate a ploidy-aware genetic relationship matrix (GRM) based on the methods described in 
Bell et al. (2017) and VanRaden et al. (2008).

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `ploidy::Int64`: Number of chromosome copies in the organism (default: 2)
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to select specific entries (default: nothing)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to select specific loci/alleles (default: nothing)
- `verbose::Bool`: If true, displays a heatmap of the resulting GRM (default: false)

# Returns
- `Matrix{Float64}`: A symmetric genetic relationship matrix with dimensions (n × n), where n is the number of entries

# Details
The function implements the following steps:
1. Extracts and processes genomic data
2. Calculates allele frequencies and centers the data
3. Computes the GRM using VanRaden's method
4. Ensures matrix invertibility by adding small values to the diagonal if necessary

# Note
The diagonal elements may be slightly inflated to ensure matrix invertibility for downstream analyses.

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
        ϵ *= 10
    end
    # Output
    if verbose
        UnicodePlots.heatmap(GRM_VanRaden)
    end
    GRM_VanRaden
end
