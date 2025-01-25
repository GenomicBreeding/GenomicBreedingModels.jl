function grmsimple(
    genomes::Genomes;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    verbose::Bool = false
)::Matrix{Float64}
    # genomes = GBCore.simulategenomes(); idx_entries = nothing; idx_loci_alleles = nothing; verbose = true;
    # Check arguments while extracting the allele frequencies but first create a dummy phenomes struct
    phenomes_dummy = Phenomes(n=length(genomes.entries), t=1)
    phenomes_dummy.entries = genomes.entries
    phenomes_dummy.populations = genomes.populations
    phenomes_dummy.traits = ["dummy_trait"]
    phenomes_dummy.phenotypes[:, 1] = rand(length(phenomes_dummy.entries))
    G, _, entries, populations, loci_alleles = extractxyetc(genomes, phenomes_dummy, idx_entries=idx_entries, idx_loci_alleles=idx_loci_alleles, add_intercept = false)
    # Calculate a simple GRM
    GRM = G * G' ./ size(G, 2)
    if verbose
        UnicodePlots.heatmap(GRM)
    end
    GRM
end

function grmploidyaware(
    genomes::Genomes;
    ploidy::Int64 = 2;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    verbose::Bool = false
)::Matrix{Float64}
    # genomes = GBCore.simulategenomes(); ploidy = 2; idx_entries = nothing; idx_loci_alleles = nothing; verbose = true;
    # Check arguments while extracting the allele frequencies but first create a dummy phenomes struct
    phenomes_dummy = Phenomes(n=length(genomes.entries), t=1)
    phenomes_dummy.entries = genomes.entries
    phenomes_dummy.populations = genomes.populations
    phenomes_dummy.traits = ["dummy_trait"]
    phenomes_dummy.phenotypes[:, 1] = rand(length(phenomes_dummy.entries))
    G, _, entries, populations, loci_alleles = extractxyetc(genomes, phenomes_dummy, idx_entries=idx_entries, idx_loci_alleles=idx_loci_alleles, add_intercept = false)
    # Calculate GRM via Bell et al (2017) and VanRaden et al (2008)
    q = mean(G, dims=1)
    G_star = ploidy .* (G .- 0.5)
    q_star = ploidy .* (q .- 0.5)
    Z = G_star .- q
    GRM_VanRaden = (Z * Z') ./ (ploidy * sum(q .* (1 .- q)))
    if verbose
        UnicodePlots.heatmap(GRM_VanRaden)
    end
    GRM
end