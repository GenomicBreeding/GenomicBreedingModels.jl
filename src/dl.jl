"""
    mlp(;
        genomes::Genomes,
        phenomes::Phenomes,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        n_layers::Int64 = 3,
        activation::Function = tanh,
        max_n_nodes::Int64 = 256,
        n_nodes_droprate::Float64 = 0.50,
        dropout_droprate::Float64 = 0.25,
        n_epochs::Int64 = 100_000,
        use_cpu::Bool = false,
        seed::Int64 = 123,
        verbose::Bool = false,
    )::Fit

Fit a genomic prediction model using a multi-layer perceptron (MLP).

# Example
```jldoctest; setup = :(using GBCore, GBModels, Suppressor)
julia> genomes = GBCore.simulategenomes(l=1_000, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);;

julia> phenomes = extractphenomes(trials);

julia> fit = Suppressor.@suppress mlp(genomes=genomes, phenomes=phenomes, n_epochs=10, use_cpu=true, verbose=false);

julia> fit.metrics["cor"] < 0.5
true
```
"""
function mlp(;
    genomes::Genomes,
    phenomes::Phenomes,
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    idx_trait::Int64 = 1,
    n_layers::Int64 = 3,
    activation::Function = tanh,
    max_n_nodes::Int64 = 256,
    n_nodes_droprate::Float64 = 0.50,
    dropout_droprate::Float64 = 0.25,
    n_epochs::Int64 = 100_000,
    use_cpu::Bool = false,
    seed::Int64 = 123,
    verbose::Bool = false,
)::Fit
    # genomes = GBCore.simulategenomes()
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing; idx_trait = 1; add_intercept = true
    # n_layers = 4
    # activation = tanh
    # max_n_nodes = 256
    # n_nodes_droprate = 0.50
    # dropout_droprate = 0.25
    # n_epochs = 10
    # use_cpu = true
    # seed = 123
    # verbose = true
    # Set the random seed
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    # Check arguments and extract X, y, and loci-allele names
    X, y, entries, populations, loci_alleles = extractxyetc(
        genomes,
        phenomes,
        idx_entries = idx_entries,
        idx_loci_alleles = idx_loci_alleles,
        idx_trait = idx_trait,
        add_intercept = false,
    )
    # Instantiate output Fit
    fit = Fit(n = size(X, 1), l = size(X, 2))
    fit.model = "multi-layer perceptron"
    fit.b_hat_labels = vcat(["intercept"], loci_alleles)
    fit.trait = phenomes.traits[idx_trait]
    fit.entries = entries
    fit.populations = populations
    fit.y_true = y
    # Construct the layers
    n, p = size(X)
    model = if n_layers == 1
        Chain(Dense(p, 1, tanh))
    elseif n_layers == 2
        Chain(Dense(p, max_n_nodes, tanh), Dense(max_n_nodes, 1, tanh))
    else
        model = Chain(Dense(p, max_n_nodes, activation), Dropout(dropout_droprate))
        for i = 2:(n_layers-1)
            in_dims = model.layers[end-1].out_dims
            out_dims = Int64(maximum([round(in_dims * n_nodes_droprate), 1]))
            dp = model.layers[end].p * dropout_droprate
            model = Chain(model, Dense(in_dims, out_dims, activation), Dropout(dp))
        end
        in_dims = model.layers[end-1].out_dims
        model = Chain(model, Dense(in_dims, 1, activation))
        model
    end
    # Get the device determined by Lux
    dev = if use_cpu
        cpu_device()
    else
        gpu_device()
    end
    # Move the data to the device (Note that we have to transpose X and y)
    x = dev(X')
    y = dev(reshape(y, 1, n))
    # Parameter and State Variables
    ps, st = Lux.setup(rng, model) |> dev
    ## First construct a TrainState
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(0.0001f0))
    ### Train
    for iter = 1:n_epochs
        _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), MSELoss(), (x, y), train_state)
        if verbose
            println("Iteration: $iter\tLoss: $loss")
        end
    end
    # Metrics
    y_pred, st = Lux.apply(model, x, ps, st)
    y_pred = y_pred[1, :]
    y = y[1, :]
    performance = metrics(y, y_pred)
    if verbose
        @show performance
        @show UnicodePlots.scatterplot(y, y_pred)
    end
    # Output
    fit.b_hat = zeros(length(fit.b_hat_labels))
    fit.lux_model = model
    fit.y_pred = y_pred
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end
