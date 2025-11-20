"""
    extractxyetc(
        genomes::Genomes,
        phenomes::Phenomes;
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        add_intercept::Bool = true
    ) -> Tuple{Matrix{Float64}, Vector{Float64}, Vector{String}, Vector{String}, Vector{String}}

Extract data matrices and vectors from genomic and phenotypic data for statistical analyses.

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to select specific entries (default: all entries)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to select specific loci-alleles (default: all loci-alleles)
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `add_intercept::Bool`: Whether to add an intercept column to the design matrix (default: true)

# Returns
A tuple containing:
1. `X::Matrix{Float64}`: Design matrix with allele frequencies (and intercept if specified)
2. `y::Vector{Float64}`: Response vector with phenotypic values
3. `entries::Vector{String}`: Names of the selected entries
4. `populations::Vector{String}`: Population identifiers for the selected entries
5. `loci_alleles::Vector{String}`: Names of the selected loci-alleles

# Notes
- The function filters out entries with missing, NaN, or infinite phenotype values
- Requires at least 2 valid entries after filtering
- Checks for non-zero variance in the trait values
- Ensures consistency between genomic and phenotypic data dimensions
- Validates all index inputs are within bounds

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false);

julia> trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> X, y, loci_alleles = extractxyetc(genomes, phenomes);

julia> X == hcat(ones(length(phenomes.entries)), genomes.allele_frequencies)
true

julia> y == phenomes.phenotypes[:, 1]
true
```
"""
function extractxyetc(
    genomes::Genomes,
    phenomes::Phenomes;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    add_intercept::Bool = true,
)::Tuple{Matrix{Float64},Vector{Float64},Vector{String},Vector{String},Vector{String}}
    # genomes = GenomicBreedingCore.simulategenomes()
    # trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing
    # idx_trait = 1; add_intercept = true
    # Check arguments
    if !checkdims(genomes) && !checkdims(phenomes)
        throw(ArgumentError("The Genomes and Phenomes structs are corrupted ☹."))
    end
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted ☹."))
    end
    if !checkdims(phenomes)
        throw(ArgumentError("The Phenomes struct is corrupted ☹."))
    end
    if genomes.entries != phenomes.entries
        throw(ArgumentError("The genomes and phenomes input need to have been merged to have consitent entries."))
    end
    if isnothing(idx_entries)
        idx_entries = collect(1:length(genomes.entries))
    else
        if (minimum(idx_entries) < 1) .|| maximum(idx_entries) > length(genomes.entries)
            throw(
                ArgumentError(
                    "The indexes of the entries, `idx_entries` are out of bounds. Expected range: from 1 to " *
                    string(length(genomes.entries)) *
                    " while the supplied range is from " *
                    string(minimum(idx_entries)) *
                    " to " *
                    string(maximum(idx_entries)) *
                    ".",
                ),
            )
        end
    end
    if isnothing(idx_loci_alleles)
        idx_loci_alleles = collect(1:length(genomes.loci_alleles))
    else
        if (minimum(idx_loci_alleles) < 1) .|| maximum(idx_loci_alleles) > length(genomes.loci_alleles)
            throw(
                ArgumentError(
                    "The indexes of the loci_alleles, `idx_loci_alleles` are out of bounds. Expected range: from 1 to " *
                    string(length(genomes.loci_alleles)) *
                    " while the supplied range is from " *
                    string(minimum(idx_loci_alleles)) *
                    " to " *
                    string(maximum(idx_loci_alleles)) *
                    ".",
                ),
            )
        end
    end
    # Extract the response variable
    ϕ = phenomes.phenotypes[idx_entries, idx_trait]
    # Omit entries missing, nan and infinite phenotype data
    idx::Vector{Int64} = findall(.!ismissing.(ϕ) .&& .!isnan.(ϕ) .&& .!isinf.(ϕ))
    if length(idx) < 2
        throw(
            ArgumentError(
                "There are less than 2 entries with non-missing phenotype data after merging with the genotype data.",
            ),
        )
    end
    y::Vector{Float64} = ϕ[idx]
    if var(y) < 1e-20
        throw(ErrorException("Very low or zero variance in trait: `" * phenomes.traits[idx_trait] * "`."))
    end
    # Extract the explanatory matrix
    G::Matrix{Float64} = genomes.allele_frequencies[idx_entries[idx], idx_loci_alleles]
    # Output X with/without intercept, y, names of the entries, populations, and loci-alleles
    entries = genomes.entries[idx_entries[idx]]
    populations = genomes.populations[idx_entries[idx]]
    loci_alleles = genomes.loci_alleles[idx_loci_alleles]
    if add_intercept
        return (hcat(ones(length(idx)), G), y, entries, populations, loci_alleles)
    else
        return (G, y, entries, populations, loci_alleles)
    end
end


"""
    predict(; fit::Fit, genomes::Genomes, idx_entries::Vector{Int64})::Vector{Float64}

Calculate predicted phenotypes using a fitted genomic prediction model.

# Arguments
- `fit::Fit`: A fitted genomic prediction model containing coefficients and model information
- `genomes::Genomes`: Genomic data containing allele frequencies
- `idx_entries::Vector{Int64}`: Vector of indices specifying which entries to predict

# Returns
- `Vector{Float64}`: Predicted phenotypic values for the specified entries

# Details
Supports various linear genomic prediction models including:
- OLS (Ordinary Least Squares)
- Ridge Regression
- LASSO
- Bayes A
- Bayes B
- Bayes C

The function validates input dimensions and compatibility between the fitted model and genomic data
before making predictions.

# Throws
- `ArgumentError`: If the Fit or Genomes structs are corrupted
- `ArgumentError`: If entry indices are out of bounds
- `ArgumentError`: If loci-alleles in the fitted model don't match the validation set
- `ArgumentError`: If the genomic prediction model is not recognized

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, StatsBase)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false);

julia> trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = extractphenomes(trials);

julia> fit = ridge(genomes=genomes, phenomes=phenomes, idx_entries=collect(1:90));

julia> y_hat = GenomicBreedingModels.predict(fit=fit, genomes=genomes, idx_entries=collect(91:100));

julia> cor(phenomes.phenotypes[91:100, 1], y_hat) > 0.5
true
```
"""
function predict(; fit::Fit, genomes::Genomes, idx_entries::Vector{Int64})::Vector{Float64}
    # genomes = GenomicBreedingCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
    # trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
    # phenomes = extractphenomes(trials)
    # fit = ridge(genomes=genomes, phenomes=phenomes, idx_entries=collect(1:200))
    # idx_entries = collect(201:300)
    # Check arguments
    if !checkdims(fit)
        throw(ArgumentError("The Fit struct is corrupted ☹."))
    end
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted ☹."))
    end
    if (minimum(idx_entries) < 1) .|| maximum(idx_entries) > length(genomes.entries)
        throw(
            ArgumentError(
                "The indexes of the entries, `idx_entries` are out of bounds. Expected range: from 1 to " *
                string(length(genomes.entries)) *
                " while the supplied range is from " *
                string(minimum(idx_entries)) *
                " to " *
                string(maximum(idx_entries)) *
                ".",
            ),
        )
    end
    idx_loci_alleles = try
        [findall(genomes.loci_alleles .== x)[1] for x in fit.b_hat_labels[2:end]] # errors if at least one locus in model fit is not found in genomes struct
    catch
        throw(
            ArgumentError(
                "The loci-alleles in the fitted genomic prediction model do not match the loci-alleles in the requested validation set.\nThe genomes struct can have more loci-alleles than the fitted model, but all the loci-alleles in the fitted model should be in the genomes struct. Make sure the loci-alleles of the genomes struct were not filtered.",
            ),
        )
    end
    # Linear or non-linear model prediction
    linear_models = ["ols", "ridge", "lasso", "bayesa", "bayesb", "bayesc"]
    non_linear_models = [""]
    if sum(linear_models .== fit.model) == 1
        return fit.b_hat[1] .+ (genomes.allele_frequencies[idx_entries, idx_loci_alleles] * fit.b_hat[2:end])
    elseif sum(non_linear_models .== fit.model) == 1
        # TODO: once we have non-linear models
        return NaN
    else
        throw(ArgumentError("Unrecognised genomic prediction model: `" * fit.model * "`."))
    end
end
