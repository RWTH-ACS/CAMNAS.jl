#=
Author: Pascal Bauer <pascal.bauer@rwth-aachen.de>
SPDX-FileCopyrightText: 2025 Pascal Bauer <pascal.bauer@rwth-aachen.de>
=#

"""
This module adds functions to generate LR-decomposable matrices with specifiy density. 

"""
module Generator

using Random
using SparseMatricesCSR
using LinearAlgebra
using Base

"""
    Settings

A struct that encapsulates the parameters for generating a diagonally dominant matrix.
*See 'generate_matrix(settings::Settings)'*

# Fields
- `dimension::UInt`: The size of the square matrix (number of rows and columns).
- `density::Float64`: The proportion of non-zero off-diagonal elements in each row. For example, a density of `0.1` means approximately 10% of the off-diagonal elements in each row will be non-zero.
- `magnitude_off::Float64`: The range of random values for off-diagonal elements, where values are sampled from `[-magnitude_off, magnitude_off]` (default: `0.05`).
- `delta::Float64`: A small positive value added to the diagonal to ensure diagonal dominance (default: `0.5`).
- `seed::UInt`: The random seed for reproducibility (default: a randomly generated seed).

# Example
```julia
using Generator

settings = Generator.Settings(
    dimension=5,
    density=0.1,
    magnitude_off=0.1,
    delta=1.0,
    seed=42
)
```
"""
@kwdef struct Settings
    dimension::UInt
    density::Float64
    magnitude_off::Float64 = 0.05
    delta::Float64 = 0.5
    seed::UInt = rand(UInt)
end

"""
    generate_matrix(settings::Settings)

Generates a diagonally dominant square matrix based on the given `Settings`.

# Arguments
- `settings::Settings`: A struct containing the parameters for matrix generation:
    - `dimension::UInt`: The size of the square matrix.
    - `density::Float64`: The proportion of non-zero off-diagonal elements.
    - `magnitude_off::Float64`: The range of random values for off-diagonal elements.
    - `delta::Float64`: A small positive value added to the diagonal.
    - `seed::UInt`: The random seed for reproducibility.

# Returns
- `Matrix{Float64}`: A square matrix of size `settings.dimension x settings.dimension` with the specified properties.

# Details
- The matrix is constructed to be diagonally dominant, meaning the absolute value of each diagonal element is greater than the sum of the absolute values of the off-diagonal elements in the same row.
- The density parameter controls the sparsity of the matrix. For example, a density of `0.1` means approximately 10% of the off-diagonal elements in each row will be non-zero.

# Example
```julia
using Generator

settings = Generator.Settings(
    dimension=5,
    density=0.1,
    magnitude_off=0.1,
    delta=1.0,
    seed=42
)

matrix = Generator.generate_matrix(settings)
println(matrix)
```
"""
function generate_matrix(settings::Settings)
    Random.seed!(settings.seed)
    matrix = zeros(Float64, settings.dimension, settings.dimension)

    for i in 1:settings.dimension
        s = max(1, Int(round(settings.density * (settings.dimension - 1))))
        cols = collect(1:settings.dimension)
        deleteat!(cols, i) # remove diagonal index
        cols = randperm(length(cols))[1:s] # pick s random off-diagonal columns
        for j in cols
            matrix[i, j] = rand() * 2settings.magnitude_off - settings.magnitude_off
        end

        # diagonal dominance
        row_sum = sum(abs.(matrix[i, :]))
        matrix[i, i] = row_sum + settings.delta
    end

    return matrix
end


"""
    generate_rhs_vector(matrix::Matrix{Float64}; prefered_solution::Vector{Float64}=ones(Float64, size(matrix, 1)))

Generates a right-hand side (RHS) vector for a given matrix and a preferred solution.

# Arguments
- `matrix::Matrix{Float64}`: The input matrix for which the RHS vector is generated.
- `prefered_solution::Vector{Float64}`: The preferred solution vector (default: a vector of ones).

# Returns
- `Vector{Float64}`: The RHS vector computed as `matrix * prefered_solution`.

# Example
```julia
using Generator

matrix = Generator.generate_matrix(5; density=0.1, seed=42)
rhs_vector = Generator.generate_rhs_vector(matrix)
```
"""
function generate_rhs_vector(matrix::Matrix{Float64}; prefered_solution::Vector{Float64}=ones(Float64, size(matrix, 1)))
    rhs_vector = matrix * prefered_solution
    return rhs_vector
end


"""
    matrix_to_file(csr::SparseMatrixCSR; matrix_path="\$(@__DIR__)/system_matrix_generated.txt")

Saves a sparse matrix in CSR format to a file.

# Arguments
- `csr::SparseMatrixCSR`: The sparse matrix in CSR format to be saved.
- `matrix_path::AbstractString`: The file path where the matrix will be saved (default: `system_matrix_generated.txt`).

# Details
- The matrix is saved in the following format:
  - First line: Non-zero values (`nzval`).
  - Second line: Row pointers (`rowptr`).
  - Third line: Column indices (`colval`).
  - Fourth line: Number of rows (`m`).
  - Fifth line: Number of non-zero elements.

# Example
```julia
using Generator

matrix = Generator.generate_matrix(5; density=0.1, seed=42)
csr_matrix = SparseMatricesCSR.SparseMatrixCSR(matrix)
Generator.matrix_to_file(csr_matrix, matrix_path="matrix.txt")
```
"""
function matrix_to_file(csr::SparseMatrixCSR; matrix_path="$(@__DIR__)/system_matrix_generated.txt")
    mkpath(dirname(matrix_path))
    io = open(matrix_path, "w")

    type_stripped_nzval = isempty(csr.nzval) ? "[]" : csr.nzval
    type_stripped_colval = isempty(csr.colval) ? "[]" : csr.colval 

    write(io, "$type_stripped_nzval\n$(csr.rowptr)\n$(type_stripped_colval)\n$(csr.m)\n$(length(csr.nzval))")
    close(io)
end

"""
    rhs_to_file(rhs_vector::Vector{Float64}; rhs_path="\$(@__DIR__)/rhs_generated.txt")

Saves a right-hand side (RHS) vector to a file.

# Arguments
- `rhs_vector::Vector{Float64}`: The RHS vector to be saved.
- `rhs_path::AbstractString`: The file path where the RHS vector will be saved (default: `rhs_generated.txt`).

# Example
```julia
using Generator

matrix = Generator.generate_matrix(5; density=0.1, seed=42)
rhs_vector = Generator.generate_rhs_vector(matrix)
Generator.rhs_to_file(rhs_vector, rhs_path="rhs.txt")
```
"""
function rhs_to_file(rhs_vector::Vector{Float64}; rhs_path="$(@__DIR__)/rhs_generated.txt")
    mkpath(dirname(rhs_path))

    io = open(rhs_path, "w")
    write(io, "$(rhs_vector)")
    close(io)
end

end