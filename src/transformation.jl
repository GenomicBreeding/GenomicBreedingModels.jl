"""
    transform1(
        f::Function,
        genomes::Genomes,
        phenomes::Phenomes;
        idx_trait::Int64 = 1,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        ϵ::Float64 = eps(Float64),
        use_abs::Bool = false,
    )::Genomes

Apply a function to each allele frequency in genomes.
Please Use named functions if you wish to reconstruct the transformation from the `loci_alleles` field.

# Example
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(verbose=false);

julia> genomes_transformed = transform1(x -> x^2, genomes);

julia> mean(sqrt.(genomes_transformed.allele_frequencies) .- genomes.allele_frequencies) < 1e-10
true

julia> squareaddpi(x) = x^2 + pi;

julia> genomes_transformed = transform1(squareaddpi, genomes);

julia> mean(squareaddpi.(genomes.allele_frequencies[:,1]) .- genomes_transformed.allele_frequencies[:,1]) < 1e-10
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
    n_new_features_per_transformation::Int64 = 1_000
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
    X, _, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = 1, # the phenotypes are unimportant at this stage
        add_intercept = false,
    )
    # Add the epsilon
    X .+= ϵ
    # Use the absolute values if requested
    if use_abs
        X = abs.(X)
    end
    # Instantiate the output genomes
    out = Genomes(n = size(X, 1), p = n_new_features_per_transformation)
    out.entries = entries
    out.populations = populations
    # Apply the transformation iteratively so that we ddo not run our of memory
    β = zeros(size(X, 2))
    g = Genomes(n=size(X,1), p=1); g.entries = genomes.entries; g.populations = genomes.populations
    for j in eachindex(β)
        # j = 1
        x = X[:, j]
        if var(x) < σ²_threshold
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
        fit = ols(genomes=g, phenomes=phenomes, idx_trait=idx_trait)
        β[j] = fit.b_hat[2]
    end
    idx = sortperm(β, rev=true)[1:n_new_features_per_transformation]
    # Output
    out.allele_frequencies = genomes.allele_frequencies[:, idx]
    out.loci_alleles = string.(f, "(", genomes.loci_alleles[idx], ")")
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
        ϵ::Float64 = eps(Float64),
        use_abs::Bool = false,
    )::Genomes

Apply a function to pairs of allele frequency in genomes.
Please Use named functions if you wish to reconstruct the transformation from the `loci_alleles` field.

# Example
```jldoctest; setup = :(using GBCore, GBModels)
julia> genomes = GBCore.simulategenomes(l=1_000, verbose=false);

julia> genomes_transformed = transform2((x,y) -> (x^2 + sqrt(y)) / 2, genomes);

julia> mean((genomes.allele_frequencies[:,1].^2 .+ sqrt.(genomes.allele_frequencies[:,2])) ./ 2 .- genomes_transformed.allele_frequencies[:,2]) < 1e-10
true

julia> raisexbyythenlog(x, y) = log(abs(x^y));

julia> genomes_transformed = transform2(raisexbyythenlog, genomes);

julia> mean(raisexbyythenlog.(genomes.allele_frequencies[:,1], genomes.allele_frequencies[:,2]) .- genomes_transformed.allele_frequencies[:,2]) < 1e-10
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
    commutative::Bool = false,
    ϵ::Float64 = eps(Float64),
    use_abs::Bool = false,
    verbose::Bool = false,
)::Genomes
    # f=(x,y) -> (x^2 + sqrt(y)) / 2; genomes = GBCore.simulategenomes(l=100, verbose=false); idx_entries = nothing; idx_loci_alleles = nothing; commutative = false; ϵ = eps(Float64); use_abs = true; verbose = true;
    # Check arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    # Define a dummy phenomes struct (we do not need phenotype information here)
    dummy_phenomes = Phenomes(n = length(genomes.entries), t = 1)
    dummy_phenomes.entries = genomes.entries
    dummy_phenomes.populations = genomes.populations
    dummy_phenomes.phenotypes[:, 1] = rand(length(genomes.entries))
    # Extract the allele frequencies matrix, etc, while checking the rest of the arguments
    X, _, entries, populations, loci_alleles = extractxyetc(
        genomes,
        dummy_phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = 1, # the phenotypes are unimportant at this stage
        add_intercept = false,
    )
    # Add the epsilon
    X .+= ϵ
    # Use the absolute values if requested
    if use_abs
        X = abs.(X)
    end
    # Instantiate the output matrix and feature names
    # Note that the transformation will be applied to the diagonal and the upper trianular of pairs if the function is commutative, 
    # otherwise all possible ordered pairs will be used.
    n, p = size(X)
    k::Int64 = if commutative
        ((p^2 - p) / 2) + p
    else
        p^2
    end
    if verbose
        println("Will be generating a transformed matrix with " * string(n) * " rows and " * string(k) * " columns.")
    end
    T = fill(0.0, n, k)
    features = fill("", k)
    # Apply the transformation
    if verbose
        pb = Progress(k; desc = "Applying tranformation to generate " * string(k) * " new features: ")
    end
    counter = 0
    for i = 1:p
        range_2nd = if commutative
            i:p
        else
            1:p
        end
        for j in range_2nd
            # i = 2; j = 10;
            counter += 1
            T[:, counter] = try
                f.(X[:, i], X[:, j])
            catch
                throw(
                    ArgumentError(
                        "Cannot transform the allele frequencies using the function: `" *
                        string(f) *
                        "`. Please make sure that this function accepts two arguments. Also, please consider adding a larger `ϵ` (currently equal to " *
                        string(ϵ) *
                        ") and/or using absolute values, i.e. use `use_abs=true` (currently equal to " *
                        string(use_abs) *
                        ").",
                    ),
                )
            end
            features[counter] = string(f, "(", loci_alleles[i], ",", loci_alleles[j], ")")
            if verbose
                next!(pb)
            end
        end
    end
    if verbose
        finish!(pb)
    end
    # Output
    out = Genomes(n = n, p = k)
    out.entries = entries
    out.populations = populations
    out.allele_frequencies = T
    out.loci_alleles = features
    if !checkdims(out)
        throw(ErrorException("Error transforming pairs of loci using the function `" * string(f) * "`."))
    end
    out
end

sqrtabs(x) = sqrt(abs(x))
log10abseps(x) = log10(abs(x) + eps(Float64))
inveps(x) = 1/(x+eps(Float64))

mult(x,y) = x * y
add(x,y) = x + y
raise(x,y) = x ^ y

function epistasisfeatures(
    genomes::Genomes,
    phenomes::Phenomes;
    idx_trait::Int64 = 1,
    transformations1::Vector{Function} = [sqrtabs, log10abseps, inveps],
    transformations2::Vector{Function} = [mult, add, raise],
    n_new_features_per_transformation::Int64 = 1_000,
    n_reps::Int64 = 3,
)::Genomes
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials); idx_trait = 1;
    # transformations1 = [sqrtabs, log10abseps, inveps]
    # transformations2 = [mult, add, raise]
    # n_new_features_per_transformation = 100
    # n_reps = 3


    idx_entries = collect(1:length(genomes.entries))
    γ = clone(genomes)

    for r in 1:n_reps
        for t1 in transformations1
            # t1 = transformations1[1]
            g = transform1(t1, γ)
            fit = lasso(genomes=g, phenomes=phenomes, idx_trait=idx_trait)
            idx = sortperm(abs.(fit.b_hat[2:end]), rev=true)[1:n_new_features_per_transformation]
            idx_loci_alleles = [findall(g.loci_alleles .== x)[1] for x in fit.b_hat_labels[2:end][idx]]
            g = slice(g, idx_entries=idx_entries, idx_loci_alleles=idx_loci_alleles)
            @show dimensions(g);
            idx_new_loci_alleles = [findall(g.loci_alleles .== x)[1] for x in setdiff(g.loci_alleles, γ.loci_alleles)]
            append!(γ.loci_alleles, g.loci_alleles[idx_new_loci_alleles])
            γ.allele_frequencies = hcat(γ.allele_frequencies, g.allele_frequencies[:, idx_new_loci_alleles])
            γ.mask = hcat(γ.mask, g.mask[:, idx_new_loci_alleles])
            @show dimensions(γ);
        end
        for t2 in transformations2
            # t2 = transformations2[1]
            g = transform2(t2, γ)
            fit = lasso(genomes=g, phenomes=phenomes, idx_trait=idx_trait)
            idx = sortperm(abs.(fit.b_hat[2:end]), rev=true)[1:n_new_features_per_transformation]
            idx_loci_alleles = [findall(g.loci_alleles .== x)[1] for x in fit.b_hat_labels[2:end][idx]]
            g = slice(g, idx_entries=idx_entries, idx_loci_alleles=idx_loci_alleles)
            @show dimensions(g);
            idx_new_loci_alleles = [findall(g.loci_alleles .== x)[1] for x in setdiff(g.loci_alleles, γ.loci_alleles)]
            append!(γ.loci_alleles, g.loci_alleles[idx_new_loci_alleles])
            γ.allele_frequencies = hcat(γ.allele_frequencies, g.allele_frequencies[:, idx_new_loci_alleles])
            γ.mask = hcat(γ.mask, g.mask[:, idx_new_loci_alleles])
            @show dimensions(γ);
        end
    end

end

function reconstitutefeatures()
end