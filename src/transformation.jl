# Transformations which maps into the same zero to one domain
"""
    square(x) = x^2

This function takes a number and squares it.
A member of the set of endofunctions defined in GBModels.jl for building non-linear genomic prediction models.
Both input and ouput range between zero and one.
"""
square(x) = x^2

"""
    invoneplus(x) = 1 / (1 + x)

This function takes a number, adds one and inverts it.
A member of the set of endofunctions defined in GBModels.jl for building non-linear genomic prediction models.
Both input and ouput range between zero and one.
"""
invoneplus(x) = 1 / (1 + x)

"""
    log10epsdivlog10eps(x) = (log10(abs(x) + eps(Float64))) / log10(eps(Float64))

This function takes a number, adds a very tiny value, takes the log10, and divide it by the log10 of the same very tiny value.
A member of the set of endofunctions defined in GBModels.jl for building non-linear genomic prediction models.
Both input and ouput range between zero and one.
"""
log10epsdivlog10eps(x) = (log10(x + eps(Float64))) / log10(eps(Float64))

"""
    mult(x, y) = x * y

This function takes two numbers and multiplies them together.
A member of the set of endofunctions defined in GBModels.jl for building non-linear genomic prediction models.
Inputs and ouput range between zero and one.
"""
mult(x, y) = x * y

"""
    addnorm(x, y) = (x + y) / 2.0

This function takes two numbers and finds their arithmetic mean.
A member of the set of endofunctions defined in GBModels.jl for building non-linear genomic prediction models.
Inputs and ouput range between zero and one.
"""
addnorm(x, y) = (x + y) / 2.0

"""
    raise(x, y) = x^y

This function raises the first value to the power of the second value.
A member of the set of endofunctions defined in GBModels.jl for building non-linear genomic prediction models.
Inputs and ouput range between zero and one.
"""
raise(x, y) = x^y

"""
    transform1(
        f::Function,
        genomes::Genomes,
        phenomes::Phenomes;
        idx_trait::Int64 = 1,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        n_new_features_per_transformation::Int64 = 1_000,
        ϵ::Float64 = eps(Float64),
        use_abs::Bool = false,
        σ²_threshold::Float64 = 0.01,
        verbose::Bool = false
    )::Genomes

Apply a transformation function to each allele frequency in genomic data while considering their effects on phenotypes.

# Arguments
- `f::Function`: Transformation function to apply to allele frequencies
- `genomes::Genomes`: Input genomic data structure
- `phenomes::Phenomes`: Corresponding phenotypic data structure

# Keywords
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `idx_entries::Union{Nothing,Vector{Int64}}`: Indices of entries to include (default: all)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Indices of loci-alleles to include (default: all)
- `n_new_features_per_transformation::Int64`: Maximum number of transformed features to retain (default: 1000)
- `ϵ::Float64`: Small value added to prevent numerical issues (default: machine epsilon)
- `use_abs::Bool`: Whether to use absolute values before transformation (default: false)
- `σ²_threshold::Float64`: Minimum variance threshold for considering a locus (default: 0.01)
- `verbose::Bool`: Whether to show progress and plots (default: false)

# Returns
- `Genomes`: A new Genomes object containing the transformed allele frequencies

# Details
The function performs the following steps:
1. Extracts allele frequencies and phenotypic data
2. Applies the transformation function to each locus
3. Estimates effects using ordinary least squares
4. Selects the most important transformed features
5. Cleans numerical artifacts (values near 0 or 1)

# Notes
- Use named functions to ensure transformations can be reconstructed from `loci_alleles`
- The function adds `ϵ` to frequencies to prevent numerical issues
- Features with variance below `σ²_threshold` are skipped
- The output contains at most `n_new_features_per_transformation` features

# Example
```jldoctest; setup = :(using GBCore, GBModels, StatsBase, DataFrames)
julia> genomes = GBCore.simulategenomes(l=1_000, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> genomes_transformed = transform1(x -> x^2, genomes, phenomes);

julia> idx = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[1])[1];

julia> mean(sqrt.(genomes_transformed.allele_frequencies[:, 1]) .- genomes.allele_frequencies[:, idx]) < 1e-10
true

julia> squareaddpi(x) = x^2 + pi;

julia> genomes_transformed = transform1(squareaddpi, genomes, phenomes);

julia> idx = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[1])[1];

julia> mean(squareaddpi.(genomes.allele_frequencies[:, idx]) .- genomes_transformed.allele_frequencies[:, 1]) < 1e-10
true
```
"""
function transform1(
    f::Function,
    genomes::Genomes,
    phenomes::Phenomes;
    idx_trait::Int64 = 1,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    n_new_features_per_transformation::Int64 = 1_000,
    ϵ::Float64 = eps(Float64),
    use_abs::Bool = false,
    σ²_threshold::Float64 = 0.01,
    verbose::Bool = false,
)::Genomes
    # f=sqrt
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials); idx_trait = 1; idx_entries = nothing; idx_loci_alleles = nothing; 
    # n_new_features_per_transformation = 1_000; ϵ = eps(Float64); use_abs = true; σ²_threshold = 0.01; verbose = true;
    # Extract the allele frequencies matrix, etc, while checking the arguments
    X, y, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        add_intercept = false,
    )
    # Add the epsilon
    X .+= ϵ
    # Use the absolute values if requested
    if use_abs
        X = abs.(X)
    end
    # Instantiate the effects of each locus-allele combination as well as the single locus-allele genomes and the corresponding single trait phenomes for iterative OLS
    l = size(X, 2)
    β = zeros(l)
    g = Genomes(n = size(X, 1), p = 1)
    g.entries = genomes.entries
    g.populations = genomes.populations
    p = Phenomes(n = length(y), t = 1)
    p.entries = phenomes.entries
    p.populations = phenomes.populations
    p.phenotypes[:, 1] = y
    # Apply the transformation iteratively so that we do not run our of memory
    if verbose
        pb = ProgressMeter.Progress(l; desc = "Transformation of individual allele frequencies (" * string(f) * "): ")
    end
    for j in eachindex(β)
        # j = 1
        # print(j)
        x = X[:, j]
        if var(x) < σ²_threshold
            if verbose
                ProgressMeter.next!(pb)
            end
            continue
        end
        g.allele_frequencies[:, 1] = try
            f.(x)
        catch
            throw(
                ArgumentError(
                    "Cannot transform the allele frequencies using the function: `" *
                    string(f) *
                    "`. Please make sure that this function accepts a single argument. Also, please consider adding a larger `ϵ` (currently equal to " *
                    string(ϵ) *
                    ") and/or using absolute values, i.e. use `use_abs=true` (currently equal to " *
                    string(use_abs) *
                    ").",
                ),
            )
        end
        fit = ols(genomes = g, phenomes = p)
        β[j] = fit.b_hat[2]
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        UnicodePlots.histogram(β)
    end
    # Sort by increasing effects
    idx_sorted = sortperm(abs.(β), rev = true)[1:n_new_features_per_transformation]
    # Remove zero effects
    idx = []
    for j in idx_sorted
        # j = idx_sorted[1]
        if abs(β[j]) > ϵ
            append!(idx, j)
        end
    end
    # Set values below ϵ to zero and 1+ϵ to one
    T = f.(X[:, idx])
    idx_zero = findall(abs.(T) .< ϵ)
    T[idx_zero] .= 0.0
    idx_one = findall(abs.(T .- 1) .< ϵ)
    T[idx_one] .= 1.0
    # Output
    out = Genomes(n = size(X, 1), p = length(idx))
    out.entries = entries
    out.populations = populations
    out.allele_frequencies = T
    out.loci_alleles = string.(f, "(", loci_alleles[idx], ")")
    if !checkdims(out)
        throw(ErrorException("Error transforming each locus using the function `" * string(f) * "`."))
    end
    out
end


"""
    transform2(
        f::Function,
        genomes::Genomes,
        phenomes::Phenomes;
        idx_trait::Int64 = 1,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        n_new_features_per_transformation::Int64 = 1_000,
        ϵ::Float64 = eps(Float64),
        use_abs::Bool = false,
        σ²_threshold::Float64 = 0.01,
        commutative::Bool = false,
        verbose::Bool = false
    )::Genomes

Apply a transformation function to pairs of allele frequencies in genomic data while considering their effects on phenotypes.

# Arguments
- `f::Function`: Transformation function to apply to pairs of allele frequencies
- `genomes::Genomes`: Input genomic data structure
- `phenomes::Phenomes`: Corresponding phenotypic data structure
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `idx_entries::Union{Nothing,Vector{Int64}}`: Subset of entries to include (default: all)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Subset of loci-alleles to include (default: all)
- `n_new_features_per_transformation::Int64`: Maximum number of transformed features to retain (default: 1000)
- `ϵ::Float64`: Small value added to frequencies to avoid numerical issues (default: machine epsilon)
- `use_abs::Bool`: Whether to use absolute values of frequencies (default: false)
- `σ²_threshold::Float64`: Minimum variance threshold for considering loci (default: 0.01)
- `commutative::Bool`: Whether the transformation function is commutative (default: false)
- `verbose::Bool`: Whether to display progress and diagnostics (default: false)

# Returns
- `Genomes`: A new Genomes object containing the transformed features

# Details
The function performs the following steps:
1. Extracts allele frequencies and phenotypic data
2. Applies the transformation function to all possible pairs of allele freuqencies (i.e., locus-allele combinations)
3. Estimates effects using ordinary least squares
4. Selects the most important transformed features
5. Cleans numerical artifacts (values near 0 or 1)

# Notes
- Use named functions to ensure transformations can be reconstructed from `loci_alleles`
- The function adds `ϵ` to frequencies to prevent numerical issues
- Features with variance below `σ²_threshold` are skipped
- The output contains at most `n_new_features_per_transformation` features

# Examples
```jldoctest; setup = :(using GBCore, GBModels, StatsBase, DataFrames)
julia> genomes = GBCore.simulategenomes(l=1_000, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> genomes_transformed = transform2((x,y) -> (x^2 + sqrt(y)) / 2, genomes, phenomes);

julia> idx_1 = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[1])[1];

julia> idx_2 = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[2])[1];

julia> mean((genomes.allele_frequencies[:,idx_1].^2 .+ sqrt.(genomes.allele_frequencies[:,idx_2])) ./ 2 .- genomes_transformed.allele_frequencies[:,1]) < 1e-10
true

julia> raisexbyythenlog(x, y) = log(abs(x^y));

julia> genomes_transformed = transform2(raisexbyythenlog, genomes, phenomes);

julia> idx_1 = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[1])[1];

julia> idx_2 = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[2])[1];

julia> mean(raisexbyythenlog.(genomes.allele_frequencies[:,idx_1], genomes.allele_frequencies[:,idx_2]) .- genomes_transformed.allele_frequencies[:,1]) < 1e-10
true
```
"""
function transform2(
    f::Function,
    genomes::Genomes,
    phenomes::Phenomes;
    idx_trait::Int64 = 1,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    n_new_features_per_transformation::Int64 = 1_000,
    ϵ::Float64 = eps(Float64),
    use_abs::Bool = false,
    σ²_threshold::Float64 = 0.01,
    commutative::Bool = false,
    verbose::Bool = false,
)::Genomes
    # f=(x,y) -> (x^2 + sqrt(y)) / 2;
    # genomes = GBCore.simulategenomes(l=1_000)
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials); idx_trait = 1; idx_entries = nothing; idx_loci_alleles = nothing; 
    # n_new_features_per_transformation = 1_000; ϵ = eps(Float64); use_abs = true; σ²_threshold = 0.01; commutative = false; verbose = true;
    # Extract the allele frequencies matrix, etc, while checking the arguments
    X, y, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        add_intercept = false,
    )
    # Add the epsilon
    X .+= ϵ
    # Use the absolute values if requested
    if use_abs
        X = abs.(X)
    end
    # Instantiate the effects of each locus-allele combination
    n, l = size(X)
    β = zeros(l^2)
    # Instantiate the single locus-allele genomes and the corresponding single trait phenomes for iterative OLS
    g = Genomes(n = n, p = 1)
    g.entries = genomes.entries
    g.populations = genomes.populations
    p = Phenomes(n = n, t = 1)
    p.entries = phenomes.entries
    p.populations = phenomes.populations
    p.phenotypes[:, 1] = y
    # Apply the transformation iteratively so that we do not run our of memory
    if verbose
        pb = ProgressMeter.Progress(l^2; desc = "Pairwise transformation of allele frequencies (" * string(f) * "): ")
    end
    counter = 0
    for i = 1:l
        for j = 1:l
            # i = 1; j = 5
            # println(string("i=", i, "; j=", j))
            counter += 1
            if commutative && (j < i)
                if verbose
                    ProgressMeter.next!(pb)
                end
                continue
            end
            x1 = X[:, i]
            x2 = X[:, j]
            if (var(x1) < σ²_threshold) || (var(x2) < σ²_threshold)
                if verbose
                    ProgressMeter.next!(pb)
                end
                continue
            end
            g.allele_frequencies[:, 1] = try
                f.(x1, x2)
            catch
                throw(
                    ArgumentError(
                        "Cannot transform the allele frequencies using the function: `" *
                        string(f) *
                        "` at loci-alleles: " *
                        replace(loci_alleles[i], "\t" => "\\t") *
                        " and " *
                        replace(loci_alleles[j], "\t" => "\\t") *
                        ". Please make sure that this function accepts two arguments. Also, please consider adding a larger `ϵ` (currently equal to " *
                        string(ϵ) *
                        ") and/or using absolute values, i.e. use `use_abs=true` (currently equal to " *
                        string(use_abs) *
                        ").",
                    ),
                )
            end
            fit = ols(genomes = g, phenomes = p)
            β[counter] = fit.b_hat[2]
            if verbose
                ProgressMeter.next!(pb)
            end
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        @show UnicodePlots.histogram(β)
        @show UnicodePlots.scatterplot(β)
    end
    # Sort by increasing effects
    idx_sorted = sortperm(abs.(β), rev = true)[1:n_new_features_per_transformation]
    # Remove zero effects
    idx = []
    for j in idx_sorted
        # j = idx_sorted[1]
        if abs(β[j]) > ϵ
            append!(idx, j)
        end
    end
    sort!(idx)
    # Instantiate the output genomes
    out = Genomes(n = size(X, 1), p = length(idx))
    out.entries = entries
    out.populations = populations
    # Extract the transformations and its corresponding names
    T = fill(0.0, n, length(idx))
    feature_names = repeat([""], length(idx))
    if verbose
        pb = ProgressMeter.Progress(length(idx); desc = "Extracting significant features: ")
    end
    for (indexer, counter) in enumerate(idx)
        # indexer = 1; counter = idx[indexer]
        i = 1 + (Int(floor((counter - 1) / l))) # This is one of the reasons why starting indexes at zero may be better
        j = 1 + ((counter - 1) % l) # This is one of the reasons why starting indexes at zero may be better
        if j == 0
            j = l
        end
        T[:, indexer] = f.(X[:, i], X[:, j])
        feature_names[indexer] = string(f, "(", loci_alleles[i], ",", loci_alleles[j], ")")
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Set values below ϵ to zero and 1+ϵ to one
    idx_zero = findall(abs.(T) .< ϵ)
    T[idx_zero] .= 0.0
    idx_one = findall(abs.(T .- 1) .< ϵ)
    T[idx_one] .= 1.0
    # Output
    out.allele_frequencies = T
    out.loci_alleles = feature_names
    if !checkdims(out)
        throw(ErrorException("Error transforming each locus using the function `" * string(f) * "`."))
    end
    out
end

"""
    epistasisfeatures(
        genomes::Genomes,
        phenomes::Phenomes;
        idx_trait::Int64 = 1,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        transformations1::Vector{Function} = [square, invoneplus, log10epsdivlog10eps],
        transformations2::Vector{Function} = [mult, addnorm, raise],
        n_new_features_per_transformation::Int64 = 1_000,
        n_reps::Int64 = 3,
        verbose::Bool = false
    )::Genomes

Generate epistasis features by applying various transformations to genomic data.

# Arguments
- `genomes::Genomes`: Input genomic data structure
- `phenomes::Phenomes`: Input phenotypic data structure
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `idx_entries::Union{Nothing,Vector{Int64}}`: Indices of entries to include (default: all)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Indices of loci/alleles to include (default: all)
- `transformations1::Vector{Function}`: Single-input transformation functions (default: [square, invoneplus, log10epsdivlog10eps])
- `transformations2::Vector{Function}`: Two-input transformation functions (default: [mult, addnorm, raise])
- `n_new_features_per_transformation::Int64`: Number of new features to generate per transformation (default: 1_000)
- `n_reps::Int64`: Number of times to repeat the transformation process (default: 3)
- `verbose::Bool`: Whether to display progress information (default:  false)

# Returns
- `Genomes`: Enhanced genomic data structure with additional epistasis features

# Description
This function generates epistasis features by applying two types of transformations:
1. Single-input transformations (transformations1) applied to individual genomic features
2. Two-input transformations (transformations2) applied to pairs of genomic features

The transformations are repeated `n_reps` times to create a rich set of derived features.

# Notes
- Ensures all generated features are within [0,1] range
- Maintains dimensional consistency of input/output structures
- Automatically handles entry and loci/allele subsetting
- Validates input data structures before processing

# Example
```jldoctest; setup = :(using GBCore, GBModels, StatsBase, DataFrames)
julia> genomes = GBCore.simulategenomes(l=1_000, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> genomes_plus_features = epistasisfeatures(genomes, phenomes, n_new_features_per_transformation=50, n_reps=2, verbose=false);

julia> cvs, notes = cvbulk(genomes=genomes_plus_features, phenomes=phenomes, models=[ridge, lasso, bayesa], verbose=false);

julia> cvs_no_epi, notes_no_epi = cvbulk(genomes=genomes, phenomes=phenomes, models=[ridge, lasso, bayesa], verbose=false);

julia> df_across, df_per_entry = GBCore.tabularise(cvs);

julia> df_across_no_epi, df_per_entry_no_epi = GBCore.tabularise(cvs_no_epi);

julia> df_summary = combine(groupby(df_across, [:trait, :model]), [[:cor] => mean, [:cor] => std]);

julia> df_summary_no_epi = combine(groupby(df_across_no_epi, [:trait, :model]), [[:cor] => mean, [:cor] => std]);

julia> mean(df_summary.cor_mean) > mean(df_summary_no_epi.cor_mean)
true
```
"""
function epistasisfeatures(
    genomes::Genomes,
    phenomes::Phenomes;
    idx_trait::Int64 = 1,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    transformations1 = [square, invoneplus, log10epsdivlog10eps],
    transformations2 = [mult, addnorm, raise],
    n_new_features_per_transformation::Int64 = 1_000,
    n_reps::Int64 = 3,
    verbose::Bool = false,
)::Genomes
    # genomes = GBCore.simulategenomes(l=1_000)
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials); idx_trait = 1; idx_entries = nothing; idx_loci_alleles = nothing;
    # transformations1 = [square, invoneplus, log10epsdivlog10eps]
    # transformations2 = [mult, addnorm, raise]
    # n_new_features_per_transformation = 100; n_reps = 3; verbose = true
    # Check arguments
    if !checkdims(genomes) && !checkdims(phenomes)
        throw(ArgumentError("The Genomes and Phenomes structs are corrupted."))
    end
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    if !checkdims(phenomes)
        throw(ArgumentError("The Phenomes struct is corrupted."))
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
    # Instantiate the output genomes struct and the input phenomes struct including only the current selected trait
    genomes = slice(genomes, idx_entries = idx_entries, idx_loci_alleles = idx_loci_alleles)
    phenomes = slice(phenomes, idx_entries = idx_entries, idx_traits = [idx_trait])

    genomes_old = clone(genomes)
    phenomes_old = clone(phenomes)

    # Generate the new features
    if verbose
        pb = ProgressMeter.Progress(
            n_reps * (length(transformations1) + length(transformations2)),
            desc = "Generating new features: ",
        )
    end
    for r = 1:n_reps
        for f in vcat(transformations1, transformations2)
            # f = transformations1[1]
            # f = transformations2[1]
            g = if f ∈ transformations1
                transform1(
                    f,
                    genomes,
                    phenomes,
                    n_new_features_per_transformation = n_new_features_per_transformation,
                    verbose = false,
                )
            else
                transform2(
                    f,
                    genomes,
                    phenomes,
                    n_new_features_per_transformation = n_new_features_per_transformation,
                    verbose = false,
                )
            end
            idx_new_loci_alleles =
                [findall(g.loci_alleles .== x)[1] for x in setdiff(g.loci_alleles, genomes.loci_alleles)]
            append!(genomes.loci_alleles, g.loci_alleles[idx_new_loci_alleles])
            genomes.allele_frequencies = hcat(genomes.allele_frequencies, g.allele_frequencies[:, idx_new_loci_alleles])
            genomes.mask = hcat(genomes.mask, g.mask[:, idx_new_loci_alleles])
            if verbose
                ProgressMeter.next!(pb)
            end
            if (minimum(genomes.allele_frequencies) < 0.0) || (abs(maximum(genomes.allele_frequencies) - 1) > 1e-12)
                throw(
                    ErrorException(
                        "The function `" *
                        string(f) *
                        "` generates values outside the expected range of zero to one. Please replace with an appropriate transforamtion function.",
                    ),
                )
            end
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Output
    if !checkdims(genomes)
        throw(ErrorException("Error generating new features."))
    end
    genomes

end


"""
    @string2operations(x)

Convert a string containing an endofunction formulae for genomic features into a parsed Julia expressions.

# Arguments
- `x`: A string containing mathematical operations on allele frequencies

# Returns
- Parsed expression that can be evaluated in Julia
"""
macro string2operations(x)
    Meta.parse(string("$(x)"))
end

"""
    reconstitutefeatures(
        genomes::Genomes;
        feature_names::Vector{String},
        verbose::Bool = false
    )::Genomes

Reconstruct epistasis features from a Genomes struct using feature names that encode the transformations applied.

# Arguments
- `genomes::Genomes`: Input genomic data structure
- `feature_names::Vector{String}`: Vector of feature names containing encoded transformation information
- `verbose::Bool`: Whether to show progress bar during reconstruction (default: false)

# Returns
- `Genomes`: A new Genomes struct with reconstructed epistasis features

# Details
The function parses the feature names to determine which transformations were applied and reconstructs
the features by applying these transformations to the original genomic data. Feature names should 
contain the transformation operations in a format that can be parsed and evaluated.

# Throws
- `ArgumentError`: If the input Genomes struct is corrupted (invalid dimensions)
- `ErrorException`: If feature reconstruction fails

# Example
```jldoctest; setup = :(using GBCore, GBModels, StatsBase, DataFrames)
julia> genomes = GBCore.simulategenomes(l=1_000, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> genomes_epifeat = epistasisfeatures(genomes, phenomes, n_new_features_per_transformation=50, n_reps=2, verbose=false);

julia> feature_names = genomes_epifeat.loci_alleles;

julia> genomes_epifeat_reconstructed = reconstitutefeatures(genomes, feature_names=feature_names);

julia> genomes_epifeat == genomes_epifeat_reconstructed
true
```
"""
function reconstitutefeatures(genomes::Genomes; feature_names::Vector{String}, verbose::Bool = false)::Genomes
    # genomes = GBCore.simulategenomes(l=1_000, verbose=false);
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;
    # phenomes = extractphenomes(trials);
    # f1(x) = x^2;
    # f2(x,y) = (x^2 + sqrt(y)) / 2;
    # genomes_transformed = transform2(f2, transform1(f1, genomes, phenomes), phenomes);
    # feature_names = genomes_transformed.loci_alleles; verbose = true;
    # Check arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    out = Genomes(n = length(genomes.entries), p = length(feature_names))
    out.entries = genomes.entries
    out.populations = genomes.populations
    out.loci_alleles = feature_names
    if verbose
        pb = ProgressMeter.Progress(length(feature_names); desc = "Reconstituting epistasis features:")
    end
    for (j, name) in enumerate(feature_names)
        # j = 1; name = feature_names[j];
        operations = vcat(split.(split(replace(name, ")" => ""), "("), ",")...)
        idx_loci_alleles = [findall(genomes.loci_alleles .== x) for x in operations]
        q = fill(0.0, length(out.entries))
        for i in eachindex(out.entries)
            # i = 2
            x = deepcopy(feature_names[j])
            for (k, idx) in enumerate(idx_loci_alleles)
                # k = 3; idx=idx_loci_alleles[k]
                if length(idx) == 1
                    qi = genomes.allele_frequencies[i, idx[1]]
                    x = replace(x, operations[k] => string(qi))
                end
            end
            out.allele_frequencies[i, j] = @eval(@string2operations $(x))
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Output
    if !checkdims(out)
        throw(ErrorException("Error reconstituting features."))
    end
    out
end
