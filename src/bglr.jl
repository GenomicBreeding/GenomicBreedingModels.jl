try
    R"library(BGLR)"
catch
    throw(ErrorException("Please install the BGLR package in R."))
end

"""
Bayesian models using BGLR, i.e. Bayes A, Bayes B and Bayes C
"""
function bglr(;
    G::Matrix{Float64},
    y::Vector{Float64},
    model::String = ["BayesA", "BayesB", "BayesC"][1],
    response_type::String = ["gaussian", "ordinal"][1],
    n_iter::Int64 = 1_500,
    n_burnin::Int64 = 500,
    verbose::Bool = false,
)::Vector{Float64}
    # genomes = GBCore.simulategenomes(n=1_000, l=1_750, verbose=true)
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=true)
    # phenomes = extractphenomes(trials)
    # G::Matrix{Float64} = genomes.allele_frequencies
    # y::Vector{Float64} = phenomes.phenotypes[:, 1]
    # model=["BayesA", "BayesB", "BayesC"][1]; response_type = ["gaussian", "ordinal"][1]; n_iter=10_500; n_burnin=100; verbose=true
    prefix_tmp_out = string(
        model,
        "-tmp-out-",
        Int64(hash(string.(vcat(y, [model, response_type, n_iter, n_burnin, verbose])))),
        "-",
        Int64(round(rand() * 1_000_000)),
    )
    @rput(G)
    @rput(y)
    @rput(model)
    @rput(response_type)
    @rput(n_iter)
    @rput(n_burnin)
    @rput(prefix_tmp_out)
    @rput(verbose)
    R"ETA = list(MRK=list(X=G, model=model, saveEffects=FALSE))"
    R"sol = BGLR::BGLR(y=y, ETA=ETA, response_type=response_type, nIter=n_iter, burnIn=n_burnin, saveAt=prefix_tmp_out, verbose=verbose)"
    @rget(sol)
    b_hat = vcat(sol[:mu], sol[:ETA][:MRK][:b])
    # Clean-up
    files = readdir()
    rm.(files[match.(Regex(string("^", prefix_tmp_out)), files).!=nothing])
    # Output
    b_hat
end
