# using DelimitedFiles

using JSON
using .Threads

# Include simulation logic
include("simulate.jl")

# Read JSON input from stdin
json_str = read(stdin, String)
batch_data = JSON.parse(json_str)

# batch_data is Vector{Any} of length N
# Convert to Vector{Matrix{Float64}}
N = length(batch_data)
batch_matrices = [Float64.(vcat([row' for row in batch_data[i]]...)) for i in 1:N]


# Prepare output array
results = Vector{Float64}(undef, N)

# Parallel simulation
Threads.@threads for i in 1:N
    # Each disturbance matrix has shape (5, 32)
    disturbance = batch_matrices[i]
    # run simulation; catch errors if needed
    results[i] = simulate_lipomdp(disturbance)
end

# Serialize results as JSON string and print
println(JSON.json(results))





