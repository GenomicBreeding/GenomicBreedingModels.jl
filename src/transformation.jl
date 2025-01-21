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
        verbose::Bool = false,
    )::Genomes

Apply a function to each allele frequency in genomes.
Please Use named functions if you wish to reconstruct the transformation from the `loci_alleles` field.

# Example
```jldoctest; setup = :(using GBCore, GBModels, StatsBase)
julia> genomes = GBCore.simulategenomes(verbose=false);

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
    # Check arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    # Extract the allele frequencies matrix, etc, while checking the rest of the arguments
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
    β = zeros(size(X, 2))
    g = Genomes(n = size(X, 1), p = 1)
    g.entries = genomes.entries
    g.populations = genomes.populations
    p = Phenomes(n = length(y), t = 1)
    p.entries = phenomes.entries
    p.populations = phenomes.populations
    p.phenotypes[:, 1] = y
    # Apply the transformation iteratively so that we do not run our of memory
    if verbose
        pb = ProgressMeter.Progress(l^2; desc = "Transformation of individual allele frequencies: ")
    end
    for j in eachindex(β)
        # j = 1
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
    # Output
    out = Genomes(n = size(X, 1), p = length(idx))
    out.entries = entries
    out.populations = populations
    out.allele_frequencies = f.(X[:, idx])
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
        verbose::Bool = false,
    )::Genomes

Apply a function to pairs of allele frequency in genomes.
Please Use named functions if you wish to reconstruct the transformation from the `loci_alleles` field.

# Example
```jldoctest; setup = :(using GBCore, GBModels, StatsBase)
julia> genomes = GBCore.simulategenomes(l=1_000, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> genomes_transformed = transform2((x,y) -> (x^2 + sqrt(y)) / 2, genomes, phenomes);

julia> idx_1 = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[1])[1];

julia> idx_2 = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[2])[1];

julia> mean((genomes.allele_frequencies[:,idx_1].^2 .+ sqrt.(genomes.allele_frequencies[:,idx_2])) ./ 2 .- genomes_transformed.allele_frequencies[:,idx_2]) < 1e-10
true

julia> raisexbyythenlog(x, y) = log(abs(x^y));

julia> genomes_transformed = transform2(raisexbyythenlog, genomes, phenomes);

julia> idx_1 = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[1])[1];

julia> idx_2 = findall(genomes.loci_alleles .== split(split(replace(genomes_transformed.loci_alleles[1], ")" => ""), "(")[2], ",")[2])[1];

julia> mean(raisexbyythenlog.(genomes.allele_frequencies[:,idx_1], genomes.allele_frequencies[:,idx_2]) .- genomes_transformed.allele_frequencies[:,idx_2]) < 1e-10
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
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials); idx_trait = 1; idx_entries = nothing; idx_loci_alleles = nothing; 
    # n_new_features_per_transformation = 1_000; ϵ = eps(Float64); use_abs = true; σ²_threshold = 0.01; commutative = false; verbose = true;
    # Check arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    # Extract the allele frequencies matrix, etc, while checking the rest of the arguments
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
        pb = ProgressMeter.Progress(l^2; desc = "Pairwise transformation of allele frequencies: ")
    end
    counter = 0
    for i = 1:l
        for j = 1:l
            # i = 1; j = 5
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
                        replace(loci_alleles[i], "\t" => "-") *
                        " and " *
                        replace(loci_alleles[j], "\t" => "-") *
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
        i = Int(floor(counter / l))
        j = counter % l
        T[:, indexer] = f.(X[:, i], X[:, j])
        feature_names[indexer] = string(f, "(", loci_alleles[i], ",", loci_alleles[j], ")")
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Output
    out.allele_frequencies = T
    out.loci_alleles = feature_names
    if !checkdims(out)
        throw(ErrorException("Error transforming each locus using the function `" * string(f) * "`."))
    end
    out
end

sqrtabs(x) = sqrt(abs(x))
log10abseps(x) = log10(abs(x) + eps(Float64))
inveps(x) = 1 / (x + eps(Float64))

mult(x, y) = x * y
add(x, y) = x + y
raise(x, y) = x^y

function epistasisfeatures(
    genomes::Genomes,
    phenomes::Phenomes;
    idx_trait::Int64 = 1,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    transformations1::Vector{Function} = [sqrtabs, log10abseps, inveps],
    transformations2::Vector{Function} = [mult, add, raise],
    n_new_features_per_transformation::Int64 = 1_000,
    n_reps::Int64 = 3,
)::Genomes
    # genomes = GBCore.simulategenomes(l=1_000)
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials); idx_trait = 1; idx_entries = nothing; idx_loci_alleles = nothing;
    # transformations1 = [sqrtabs, log10abseps, inveps]
    # transformations2 = [mult, add, raise]
    # n_new_features_per_transformation = 100
    # n_reps = 3


    if isnothing(idx_entries)
        idx_entries = collect(1:length(genomes.entries))
    end
    if isnothing(idx_loci_alleles)
        idx_loci_alleles = collect(1:length(genomes.loci_alleles))
    end

    γ = slice(genomes, idx_entries=idx_entries, idx_loci_alleles=idx_loci_alleles)
    ϕ = slice(phenomes, idx_entries=idx_entries, idx_traits=[idx_trait])

    for r = 1:n_reps
        for t1 in transformations1
            # t1 = transformations1[1]
            g = transform1(t1, γ, ϕ, n_new_features_per_transformation=n_new_features_per_transformation)
            idx_new_loci_alleles = [findall(g.loci_alleles .== x)[1] for x in setdiff(g.loci_alleles, γ.loci_alleles)]
            append!(γ.loci_alleles, g.loci_alleles[idx_new_loci_alleles])
            γ.allele_frequencies = hcat(γ.allele_frequencies, g.allele_frequencies[:, idx_new_loci_alleles])
            γ.mask = hcat(γ.mask, g.mask[:, idx_new_loci_alleles])
            @show dimensions(γ)
        end
        for t2 in transformations2
            # t2 = transformations2[1]
            g = transform2(t2, γ, ϕ, n_new_features_per_transformation=n_new_features_per_transformation)
            idx_new_loci_alleles = [findall(g.loci_alleles .== x)[1] for x in setdiff(g.loci_alleles, γ.loci_alleles)]
            append!(γ.loci_alleles, g.loci_alleles[idx_new_loci_alleles])
            γ.allele_frequencies = hcat(γ.allele_frequencies, g.allele_frequencies[:, idx_new_loci_alleles])
            γ.mask = hcat(γ.mask, g.mask[:, idx_new_loci_alleles])
            @show dimensions(γ)
        end
    end

end

function reconstitutefeatures() end
