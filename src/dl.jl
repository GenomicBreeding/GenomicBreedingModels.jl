"""
    mlp(;
        genomes::Genomes,
        phenomes::Phenomes,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        idx_trait::Int64 = 1,
        n_layers::Int64 = 3,
        activation::Function = relu,
        max_n_nodes::Int64 = 256,
        n_nodes_droprate::Float64 = 0.50,
        dropout_droprate::Float64 = 0.25,
        n_epochs::Int64 = 100_000,
        use_cpu::Bool = false,
        seed::Int64 = 123,
        verbose::Bool = false
    )::Fit

Fit a genomic prediction model using a multi-layer perceptron (MLP) neural network with Lux.jl.

# Arguments
- `genomes::Genomes`: Genetic information of the population
- `phenomes::Phenomes`: Phenotypic data of the population
- `idx_entries::Union{Nothing,Vector{Int64}}`: Indices of entries to include in the analysis
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Indices of loci-alleles to include
- `idx_trait::Int64`: Index of the trait to analyze
- `n_layers::Int64`: Number of hidden layers in the neural network
- `activation::Function`: Activation function for the neural network layers (default: relu)
- `max_n_nodes::Int64`: Maximum number of nodes in the first hidden layer
- `n_nodes_droprate::Float64`: Rate at which number of nodes decreases between layers
- `dropout_droprate::Float64`: Initial dropout rate for regularization, subsequent layers have proportionally decreasing rates
- `n_epochs::Int64`: Number of training epochs
- `use_cpu::Bool`: If true, forces CPU usage instead of GPU
- `seed::Int64`: Random seed for reproducibility
- `verbose::Bool`: If true, prints training progress and final metrics

# Returns
- `Fit`: A fitted model object containing:
  - `y_pred`: Predicted values
  - `y_true`: Observed values
  - `b_hat`: Model coefficients (placeholder zeros for MLP)
  - `lux_model`: The trained Lux neural network model
  - `metrics`: Performance metrics
  - Additional metadata about the model fit

# Details
The neural network architecture is constructed dynamically based on input parameters:
- For 1 layer: Direct input to output mapping
- For 2 layers: Input → max_n_nodes → output
- For 3+ layers: Input → max_n_nodes → progressively smaller layers → output
  with optional dropout between layers

Training uses the Adam optimizer with a learning rate of 0.0001 and MSE loss function.

# Example
```jldoctest; setup = :(using GBCore, GBModels, Suppressor)
julia> genomes = GBCore.simulategenomes(l=1_000, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = extractphenomes(trials);

julia> fit = Suppressor.@suppress mlp(genomes=genomes, phenomes=phenomes, n_epochs=1_000, use_cpu=true, verbose=false);

julia> fit.metrics["cor"] >= 0.2
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
    activation::Function = relu,
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
        Chain(Dense(p, 1, activation))
    elseif n_layers == 2
        Chain(Dense(p, max_n_nodes, activation), Dense(max_n_nodes, 1, activation))
    else
        model = if dropout_droprate > 0.0
            Chain(Dense(p, max_n_nodes, activation), Dropout(dropout_droprate))
        else
            Chain(Dense(p, max_n_nodes, activation))
        end
        for i = 2:(n_layers-1)
            model = if dropout_droprate > 0.0
                in_dims = model.layers[end-1].out_dims
                out_dims = Int64(maximum([round(in_dims * n_nodes_droprate), 1]))
                dp = model.layers[end].p * dropout_droprate
                Chain(model, Dense(in_dims, out_dims, activation), Dropout(dp))
            else
                in_dims = model.layers[end].out_dims
                out_dims = Int64(maximum([round(in_dims * n_nodes_droprate), 1]))
                Chain(model, Dense(in_dims, out_dims, activation))
            end
        end
        in_dims = if dropout_droprate > 0.0
            model.layers[end-1].out_dims
        else
            in_dims = model.layers[end].out_dims
        end
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
    Lux.trainmode(st) # not really required as the default is training mode, this is more for debugging with metrics calculations below
    for iter = 1:n_epochs
        _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), MSELoss(), (x, y), train_state)
        if verbose
            println("Iteration: $iter\tLoss: $loss")
        end
    end
    # Metrics
    Lux.testmode(st)
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
