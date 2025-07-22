using DelimitedFiles

# Read matrix from stdin
data = read(stdin, String)
lines = split(data, '\n')
matrix = [parse.(Float64, split(line)) for line in lines if !isempty(line)]
disturbance = Matrix(transpose(hcat(matrix...)))  # Transpose since hcat stacks columns

# Include simulation logic
include("simulate.jl")

# Run and print result
result = simulate_lipomdp(disturbance)
println(result)
