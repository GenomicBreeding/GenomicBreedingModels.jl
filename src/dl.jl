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

Train a multi-layer perceptron (MLP) model for genomic prediction.

# Arguments
- `genomes::Genomes`: A `Genomes` struct containing the genomic data.
- `phenomes::Phenomes`: A `Phenomes` struct containing the phenomic data.
- `idx_entries::Union{Nothing, Vector{Int64}}`: Indices of entries to include in the model. If `nothing`, all entries are included. Default is `nothing`.
- `idx_loci_alleles::Union{Nothing, Vector{Int64}}`: Indices of loci-alleles to include in the model. If `nothing`, all loci-alleles are included. Default is `nothing`.
- `idx_trait::Int64`: Index of the trait to predict. Default is 1.
- `n_layers::Int64`: Number of layers in the MLP. Default is 3.
- `activation::Function`: Activation function to use in the MLP. Default is `relu`.
- `max_n_nodes::Int64`: Maximum number of nodes in each layer. Default is 256.
- `n_nodes_droprate::Float64`: Drop rate for the number of nodes in each layer. Default is 0.50.
- `dropout_droprate::Float64`: Dropout rate for the layers. Default is 0.25.
- `n_epochs::Int64`: Number of training epochs. Default is 100,000.
- `use_cpu::Bool`: If `true`, use CPU for training. If `false`, use GPU if available. Default is `false`.
- `seed::Int64`: Random seed for reproducibility. Default is 123.
- `verbose::Bool`: If `true`, prints detailed progress information during training. Default is `false`.

# Returns
- `Fit`: A `Fit` struct containing the trained MLP model and performance metrics.

# Details
This function trains a multi-layer perceptron (MLP) model on genomic and phenomic data. The function performs the following steps:

1. **Set Random Seed**: Sets the random seed for reproducibility.
2. **Extract Features and Targets**: Extracts the feature matrix `X`, target vector `y`, and other relevant information from the genomic and phenomic data.
3. **Instantiate Output Fit**: Creates a `Fit` struct to store the model and results.
4. **Construct MLP Layers**: Constructs the MLP layers based on the specified number of layers, activation function, and dropout rates.
5. **Move Data to Device**: Moves the data to the appropriate device (CPU or GPU).
6. **Setup Training State**: Initializes the training state with the model parameters and optimizer.
7. **Train the Model**: Trains the MLP model for the specified number of epochs, printing progress if `verbose` is `true`.
8. **Evaluate Performance**: Evaluates the model's performance using the specified metrics.
9. **Output**: Returns the `Fit` struct containing the trained model and performance metrics.

# Notes
- The function uses the Lux library for constructing and training the MLP model.
- The `verbose` option provides additional insights into the training process by printing progress information.
- The function ensures that the trained model and performance metrics are stored in the `Fit` struct.

# Throws
- `ArgumentError`: If the `Genomes` or `Phenomes` struct is corrupted or if any of the arguments are out of range.
- `ErrorException`: If an error occurs during model training or evaluation.

# Example
```jldoctest; setup = :(using GBCore, GBModels, Suppressor)
julia> genomes = GBCore.simulategenomes(l=1_000, verbose=false);

julia> trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);

julia> phenomes = extractphenomes(trials);

julia> fit_cpu = Suppressor.@suppress mlp(genomes=genomes, phenomes=phenomes, n_epochs=1_000, use_cpu=true, verbose=false);

julia> fit_gpu = Suppressor.@suppress mlp(genomes=genomes, phenomes=phenomes, n_epochs=1_000, use_cpu=false, verbose=false);

julia> fit_cpu.metrics["cor"] >= 0.2
true

julia> fit_gpu.metrics["cor"] >= 0.2
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
    # genomes = GBCore.simulategenomes(n=500, l=1_000, n_populations=3, verbose=true)
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, f_add_dom_epi=[0.1 0.01 0.01;], verbose=false);
    # phenomes = extractphenomes(trials)
    # idx_entries = nothing; idx_loci_alleles = nothing; idx_trait = 1; add_intercept = true
    # n_layers = 4
    # activation = relu
    # max_n_nodes = 256
    # n_nodes_droprate = 0.50
    # dropout_droprate = 0.25
    # n_epochs = 10_000
    # use_cpu = false
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
    X_transposed::Matrix{Float64} = X'
    x = dev(X_transposed)
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
    ϕ_pred::Vector{Float64} = y_pred[1, :]
    ϕ_true::Vector{Float64} = y[1, :]
    performance = metrics(ϕ_true, ϕ_pred)
    if verbose
        @show performance
        @show UnicodePlots.scatterplot(ϕ_true, ϕ_pred)
    end
    # Output
    fit.b_hat = zeros(length(fit.b_hat_labels))
    fit.lux_model = model
    fit.y_pred = ϕ_pred
    fit.metrics = performance
    if !checkdims(fit)
        throw(ErrorException("Error fitting " * fit.model * "."))
    end
    fit
end
