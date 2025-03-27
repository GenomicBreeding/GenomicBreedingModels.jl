"""
    gwasprep(
        genomes::Genomes,
        phenomes::Phenomes;
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        GRM_type::String = ["simple", "ploidy-aware"][1],
        standardise::Bool = true,
        verbose::Bool = false,
    )::Tuple{Matrix{Float64},Vector{Float64},Matrix{Float64},Fit}

Prepare data matrices and structures for genome-wide association studies (GWAS).

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to subset entries (default: all entries)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to subset loci/alleles (default: all loci)
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `GRM_type::String`: Type of genetic relationship matrix to use ("simple" or "ploidy-aware") (default: "simple")
- `standardise::Bool`: Whether to standardize the data matrices (default: true)
- `verbose::Bool`: Whether to print progress information (default: false)

# Returns
A tuple containing:
- `G::Matrix{Float64}`: Standardized allele frequency matrix
- `y::Vector{Float64}`: Standardized phenotype vector  
- `GRM::Matrix{Float64}`: Genetic relationship matrix
- `fit::Fit`: Initialized Fit struct for GWAS results

# Details
- Performs data validation and preprocessing for GWAS analysis
- Removes fixed loci with no variation
- Standardizes genomic and phenotypic data if requested
- Constructs appropriate genetic relationship matrix
- Initializes output structure for association results

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, LinearAlgebra, StatsBase)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false);

julia> ploidy = 4;

julia> genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy;

julia> proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5;

julia> trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> G, y, GRM, fit = gwasprep(genomes=genomes, phenomes=phenomes);

julia> sum(abs.(mean(G, dims=1)[1,:]) .< 1e-10) == size(G, 2)
true

julia> sum(abs.(std(G, dims=1)[1,:] .- 1) .< 1e-10) == size(G, 2)
true

julia> (abs(mean(y)) < 1e-10, abs(std(y) - 1) < 1e-10)
(true, true)

julia> size(G, 1) == length(y)
true

julia> (size(G, 1), length(y)) == size(GRM)
true

julia> length(fit.entries) == length(y)
true

julia> length(fit.b_hat) == size(G, 2)
true
```
"""
function gwasprep(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    GRM_type::String = ["simple", "ploidy-aware"][1],
    standardise::Bool = true,
    verbose::Bool = false,
)::Tuple{Matrix{Float64},Vector{Float64},Matrix{Float64},Fit}
    # genomes = GenomicBreedingCore.simulategenomes(); ploidy = 4; genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy
    # proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5
    # trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing; idx_trait = 1; GRM_type = "ploidy-aware"; verbose = true
    # Check arguments while extracting the allele frequencies
    G, y, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        add_intercept = false,
    )
    if sum(["simple", "ploidy-aware"] .== GRM_type) == 0
        throw(
            ArgumentError(
                "Unrecognised `GRM_type`. Please select from:\n\t‣ " * join(["simple", "ploidy-aware"], "\n\t‣ "),
            ),
        )
    end
    # Standardise G and y and remove fixed loci-alleles
    if var(y) < eps(Float64)
        throw(ArgumentError("No variance in the trait: " * phenomes.traits[idx_trait] * "."))
    end
    v = std(G, dims = 1)[1, :]
    idx_cols = findall((v .> eps(Float64)) .&& .!ismissing.(v) .&& .!isnan.(v) .&& .!isinf.(v))
    G = G[:, idx_cols]
    loci_alleles = loci_alleles[idx_cols]
    # Extract the GRM to correct for population structure
    GRM = if GRM_type == "ploidy-aware"
        # Infer ploidy level
        ploidy = Int(round(1 / minimum(G[G.!=0.0])))
        grmploidyaware(genomes, ploidy = ploidy)
    else
        # Simple GRM
        grmsimple(genomes)
    end
    if standardise
        y = (y .- mean(y)) ./ std(y)
        G = (G .- mean(G, dims = 1)) ./ v[idx_cols]'
        GRM = (GRM .- mean(GRM, dims = 1)) ./ std(GRM, dims = 1)
    end
    # Instantiate output Fit struct
    n, l = size(G)
    fit = Fit(n = n, l = l)
    fit.model = ""
    fit.trait = phenomes.traits[idx_trait]
    fit.b_hat_labels = loci_alleles
    fit.entries = entries
    fit.populations = populations
    fit.metrics = Dict("" => 0.0)
    (G, y, GRM, fit)
end

"""
    gwasols(
        genomes::Genomes,
        phenomes::Phenomes;
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        GRM_type::String = ["simple", "ploidy-aware"][1],
        verbose::Bool = false,
    )::Fit

Perform genome-wide association study (GWAS) using ordinary least squares (OLS) regression with population structure correction.

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to subset entries (default: all entries)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to subset loci/alleles (default: all loci)
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `GRM_type::String`: Type of genetic relationship matrix to use ("simple" or "ploidy-aware") (default: "simple")
- `verbose::Bool`: Whether to display progress and plots (default: false)

# Returns
- `Fit`: A structure containing GWAS results including:
  - `model`: Model identifier ("GWAS_OLS")
  - `b_hat`: Vector of effect size estimates/t-statistics for each marker
  - Additional model information

# Details
The function implements GWAS using OLS regression while accounting for population structure
through the first principal component of the genetic relationship matrix (GRM) as a covariate.
Two types of GRM can be used: "simple" assumes diploid organisms, while "ploidy-aware"
accounts for different ploidy levels.

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, LinearAlgebra, StatsBase)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false);

julia> ploidy = 4;

julia> genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy;

julia> proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5;

julia> trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit_1 = gwasols(genomes=genomes, phenomes=phenomes, GRM_type="simple");

julia> fit_1.model
"GWAS_OLS"

julia> fit_2 = gwasols(genomes=genomes, phenomes=phenomes, GRM_type="ploidy-aware");

julia> fit_2.model
"GWAS_OLS"

julia> findall(fit_1.b_hat .== maximum(fit_1.b_hat)) == findall(fit_2.b_hat .== maximum(fit_2.b_hat))
true
```
"""
function gwasols(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    GRM_type::String = ["simple", "ploidy-aware"][1],
    verbose::Bool = false,
)::Fit
    # genomes = GenomicBreedingCore.simulategenomes(); ploidy = 4; genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy
    # proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5
    # trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing; idx_trait = 1; GRM_type = "ploidy-aware"; verbose = true
    # Check arguments while preparing the G, y, GRM, and Fit struct, vector and matrices
    G, y, GRM, fit = gwasprep(
        genomes = genomes,
        phenomes = phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        GRM_type = GRM_type,
        standardise = true,
        verbose = false,
    )
    fit.model = "GWAS_OLS"
    # Iterative GWAS
    n, l = size(G)
    E = MultivariateStats.fit(PCA, GRM; maxoutdim = 1)
    if verbose
        pb = ProgressMeter.Progress(l; desc = "GWAS via OLS using " * GRM_type * " GRM:")
    end
    thread_lock::ReentrantLock = ReentrantLock()
    Threads.@threads for j = 1:l
        # j = 1
        X = hcat(ones(n), E.proj[:, 1], G[:, j])
        Vinv = pinv(X' * X)
        b = Vinv * X' * y
        # T-distributed t-statistic
        @lock thread_lock fit.b_hat[j] = b[end] / sqrt(Vinv[end, end])
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        GenomicBreedingCore.plot(fit, TDist(length(fit.entries) - 1))
    end
    # Output
    if !checkdims(fit)
        throw(ErrorException("Error performing GWAS via OLS using the " * GRM_type * " GRM."))
    end
    fit
end

"""
    gwaslmm(
        genomes::Genomes,
        phenomes::Phenomes;
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        GRM_type::String = ["simple", "ploidy-aware"][1],
        verbose::Bool = false
    )::Fit

Perform genome-wide association analysis using a linear mixed model (LMM) approach.

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices for subsetting entries
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices for subsetting loci/alleles
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `GRM_type::String`: Type of genetic relationship matrix to use:
    - "simple": Standard GRM calculation
    - "ploidy-aware": Ploidy-adjusted GRM calculation
- `verbose::Bool`: Whether to display progress and plots (default: false)

# Returns
- `Fit`: A structure containing GWAS results including:
    - `model`: Model identifier ("GWAS_LMM")
    - `b_hat`: Vector of test statistics (z-scores) for genetic markers

# Details
The function implements a mixed model GWAS using the first principal component of the genetic 
relationship matrix (GRM) as a fixed effect to control for population structure. The model 
includes random effects for entries and uses REML estimation.

# Notes
- Handles both diploid and polyploid data through the `GRM_type` parameter
- Uses multi-threading for parallel computation of marker effects
- Includes automatic convergence retry on fitting failures
- Maximum fitting time per marker is limited to 60 seconds

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, LinearAlgebra, StatsBase, Suppressor)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false);

julia> ploidy = 4;

julia> genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy;

julia> proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5;

julia> trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit_1 = Suppressor.@suppress gwaslmm(genomes=genomes, phenomes=phenomes, GRM_type="simple");

julia> fit_1.model
"GWAS_LMM"

julia> fit_2 = Suppressor.@suppress gwaslmm(genomes=genomes, phenomes=phenomes, GRM_type="ploidy-aware");

julia> fit_2.model
"GWAS_LMM"

julia> findall(fit_1.b_hat .== maximum(fit_1.b_hat)) == findall(fit_2.b_hat .== maximum(fit_2.b_hat))
true
```
"""
function gwaslmm(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    GRM_type::String = ["simple", "ploidy-aware"][1],
    verbose::Bool = false,
)::Fit
    # genomes = GenomicBreedingCore.simulategenomes(); ploidy = 4; genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy
    # proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5
    # trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing; idx_trait = 1; GRM_type = "ploidy-aware"; verbose = true
    # Check arguments while preparing the G, y, GRM, and Fit struct, vector and matrices
    G, y, GRM, fit = gwasprep(
        genomes = genomes,
        phenomes = phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        GRM_type = GRM_type,
        standardise = true,
        verbose = false,
    )
    fit.model = "GWAS_LMM"
    # Iterative GWAS
    n, l = size(G)
    E = MultivariateStats.fit(PCA, GRM; maxoutdim = 1)
    formula = string("y ~ 1 + PC1 + x + (1|entries)")
    thread_lock::ReentrantLock = ReentrantLock()
    if verbose
        pb = ProgressMeter.Progress(l; desc = "GWAS via LMM using the first PC of the" * GRM_type * " GRM:")
    end
    Threads.@threads for j = 1:l
        # j = 1
        df = DataFrames.DataFrame(y = y, entries = fit.entries, PC1 = E.proj[:, 1], x = G[:, j])
        f = @eval(@string2formula $(formula))
        model = try
            MixedModel(f, df)
        catch
            continue
        end
        model.optsum.REML = true
        model.optsum.maxtime = 60
        try
            fit!(model, progress = false)
        catch
            try
                fit!(model, progress = false)
            catch
                continue
            end
        end
        df_BLUEs = DataFrame(coeftable(model))
        # Standard normal distributed z-statistic
        @lock thread_lock fit.b_hat[j] = df_BLUEs.z[end]
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        GenomicBreedingCore.plot(fit, Normal())
    end
    # Output
    if !checkdims(fit)
        throw(ErrorException("Error performing GWAS via LMM using the " * GRM_type * " GRM."))
    end
    fit
end

"""
    loglikreml(θ::Vector{Float64}, data::Tuple{Vector{Float64},Matrix{Float64},Matrix{Float64}})::Float64

Calculate the restricted maximum likelihood (REML) log-likelihood for a mixed linear model.

# Arguments
- `θ::Vector{Float64}`: Vector of variance components [σ²_e, σ²_u] where:
    - σ²_e is the residual variance
    - σ²_u is the genetic variance
- `data::Tuple{Vector{Float64},Matrix{Float64},Matrix{Float64}}`: Tuple containing:
    - y: Vector of phenotypic observations
    - X: Design matrix for fixed effects
    - GRM: Genomic relationship matrix

# Returns
- `Float64`: The REML log-likelihood value. Returns `Inf` if matrix operations fail.

# Details
Implements the REML log-likelihood calculation for a mixed model of the form:
y = Xβ + Zu + e
where:
- β are fixed effects
- u are random genetic effects with u ~ N(0, σ²_u * GRM)
- e are residual effects with e ~ N(0, σ²_e * I)

The function constructs the variance-covariance matrices and computes the REML transformation
to obtain the log-likelihood value used in variance component estimation.

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, LinearAlgebra, StatsBase)
julia> genomes = GenomicBreedingCore.simulategenomes(verbose=false);

julia> ploidy = 4;

julia> genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy;

julia> proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5;

julia> trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> G, y, GRM, fit = gwasprep(genomes=genomes, phenomes=phenomes);

julia> loglik = loglikreml([0.53, 0.15], (y, hcat(ones(length(y)), G[:, 1]), GRM));

julia> loglik < 100
true
"""
function loglikreml(θ::Vector{Float64}, data::Tuple{Vector{Float64},Matrix{Float64},Matrix{Float64}})::Float64
    # genomes = GenomicBreedingCore.simulategenomes(); ploidy = 4; genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy
    # proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5
    # trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);
    # phenomes = extractphenomes(trials)
    # G, y, GRM, _ = gwasprep(genomes=genomes, phenomes=phenomes)
    # X = hcat(ones(length(y)), G[:, 1])
    # θ = [0.01, 0.02]
    # Data
    y = data[1]
    X = data[2]
    GRM = data[3]
    # Note that the incidence matrix random genotype effects, Z is I
    # Residual variance-covariance matrix (assuming homoscedasticity)
    σ²_e = θ[1]
    R = σ²_e * I
    # Variance-covariance matrix of the other random effects, i.e. individual genotype effects
    σ²_u = θ[2]
    D = σ²_u * GRM
    # Total variance, i.e. variance of y
    # Since Z = I, V = (Z * D * Z') + R simply becomes:
    V = D + R
    V_inv = pinv(V)
    # REML transformation of y, i.e. find P where E[Py] = 0.0
    P = V_inv - (V_inv * X * inv(X' * V_inv * X) * X' * V_inv)
    y_REML = P * y
    # Log-likelihood
    loglik = try
        0.5 * log(det(V)) + (y' * y_REML) + log(det(X' * V_inv * X))
    catch
        Inf
    end
    loglik
end

"""
    gwasreml(
        genomes::Genomes,
        phenomes::Phenomes;
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        GRM_type::String = "simple",
        verbose::Bool = false
    )::Fit

Performs genome-wide association analysis using restricted maximum likelihood estimation (REML).

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `phenomes::Phenomes`: Phenotypic data structure containing trait measurements
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to subset entries (default: nothing)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to subset loci/alleles (default: nothing)
- `idx_trait::Int64`: Index of the trait to analyze (default: 1)
- `GRM_type::String`: Type of genetic relationship matrix to use, either "simple" or "ploidy-aware" (default: "simple")
- `verbose::Bool`: Whether to display progress and plots (default: false)

# Returns
- `::Fit`: A Fit struct containing GWAS results, including effect estimates and test statistics

# Details
Implements the REML log-likelihood calculation for a mixed model of the form:
y = Xβ + Zu + e
where:
- β are fixed effects
- u are random genetic effects with u ~ N(0, σ²_u * GRM)
- e are residual effects with e ~ N(0, σ²_e * I)

The function constructs the variance-covariance matrices and computes the REML transformation
to obtain the log-likelihood value used in variance component estimation.

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, GenomicBreedingModels, LinearAlgebra, StatsBase)
julia> genomes = GenomicBreedingCore.simulategenomes(l=1_000, verbose=false);

julia> ploidy = 4;

julia> genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy;

julia> proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5;

julia> trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit_1 = gwasreml(genomes=genomes, phenomes=phenomes, GRM_type="simple");

julia> fit_1.model
"GWAS_REML"

julia> fit_2 = gwasreml(genomes=genomes, phenomes=phenomes, GRM_type="ploidy-aware");

julia> fit_2.model
"GWAS_REML"

julia> findall(fit_1.b_hat .== maximum(fit_1.b_hat)) == findall(fit_2.b_hat .== maximum(fit_2.b_hat))
true
```
"""
function gwasreml(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    GRM_type::String = ["simple", "ploidy-aware"][1],
    verbose::Bool = false,
)::Fit
    # genomes = GenomicBreedingCore.simulategenomes(l=1_000, verbose=false); ploidy = 4; genomes.allele_frequencies = round.(genomes.allele_frequencies .* ploidy) ./ ploidy
    # proportion_of_variance = zeros(9, 1); proportion_of_variance[1, 1] = 0.5
    # trials, effects = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.05 0.00 0.00;], proportion_of_variance = proportion_of_variance, verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing; idx_trait = 1; GRM_type = "ploidy-aware"; verbose = true
    # Check arguments while preparing the G, y, GRM, and Fit struct, vector and matrices
    G, y, GRM, fit = gwasprep(
        genomes = genomes,
        phenomes = phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        GRM_type = GRM_type,
        standardise = true,
        verbose = false,
    )
    fit.model = "GWAS_REML"
    # Iterative GWAS
    # Initialise the common optimisation function and the common initial parameters
    optimreml = OptimizationFunction(loglikreml, Optimization.AutoZygote())
    θ_init = [0.5, 0.5]
    n, l = size(G)
    thread_lock::ReentrantLock = ReentrantLock()
    if verbose
        pb = ProgressMeter.Progress(l; desc = "GWAS via REML using " * GRM_type * " GRM:")
    end
    Threads.@threads for j = 1:l
        # j = 1
        X = hcat(ones(n), G[:, j])
        # Define the optimisation problem where we set the limits of the error and genotype variances to be between 0 and 1 as all data are standard normalised
        prob = OptimizationProblem(optimreml, θ_init, (y, X, GRM), lb = [eps(Float64), eps(Float64)], ub = [1.0, 1.0])
        # Optimise, i.e. REML estimation (uses the BFGS optimiser with a larger (2x) than default absolute tolerance in the gradient (Default g_tol=1e-8) for faster convergence)
        sol = solve(prob, Optim.BFGS(), g_tol = 1e-4)
        R = sol.u[1] * I
        D = sol.u[2] * GRM
        # Since Z = I, V = (Z * D * Z') + R simply becomes:
        V = D + R
        V_inv = pinv(V)
        b = pinv(X' * V_inv * X) * (X' * V_inv * y)
        σ²_b = inv(X' * V_inv * X)
        # Standard normal distributed z-statistic
        @lock thread_lock fit.b_hat[j] = b[end] / sqrt(σ²_b[end])
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        GenomicBreedingCore.plot(fit, Normal())
    end
    # Output
    if !checkdims(fit)
        throw(ErrorException("Error performing GWAS via REML using the " * GRM_type * " GRM."))
    end
    fit
end
