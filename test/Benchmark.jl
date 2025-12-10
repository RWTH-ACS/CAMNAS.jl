#=
Author: Pascal Bauer <pascal.bauer@rwth-aachen.de>
SPDX-FileCopyrightText: 2025 Pascal Bauer <pascal.bauer@rwth-aachen.de>
=#

module Benchmark

using CAMNAS

using Base
using BenchmarkTools
using SparseMatricesCSR
using SparseArrays
using CSV, DataFrames

include("Utils.jl")
using .Utils

@kwdef struct BenchmarkResult
    decomp_elapses::Float64
    solve_elapses::Float64
end

"""
    function benchmark(dpsim_matrix::dpsim_csr_matrix, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))

*Note: Camnas debug output gets disabled during this funcion for performance reasons*

This function benchmarks the currently active system configutaion by measuring lu-decomposing and solving times 
    of a given `dpsim_matrix::dpsim_csr_matrix` with a `rhs_vector::Vector{Float64}`.

# Arguments
- `dpsim_matrix::dpsim_csr_matrix`: Input matrix
- `rhs_vector::Vector{Float64}`: The right hand side of A * x = rhs
- `samples::UInt=UInt(3)`: Number of executions (only lowest time counts)

# Returns
- `BenchmarkResult`: The benchmark result containing
    - `decomp_elapses::Float64`: Duration of lu-decomposition
    - `solve_elapses::Float64`: Duration of solving

# Example
```julia
using Benchmark
using Generator

# gerator matrix
settings = Generator.Settings(
    dimension=300,
    density=0.01
)
matrix = Generator.generate_matrix(settings)
rhs_vector = Generator.generate_rhs_vector(matrix)
result =  Benchmark.benchmark(matrix, rhs_vector) 
```
"""
function benchmark(dpsim_matrix::dpsim_csr_matrix, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))
    # Save Environment
    saved_debug_env = get(ENV, "JULIA_DEBUG", "") # Use default value if not found
    ENV["JULIA_DEBUG"] = "" # Disable debug information during execution

    GC.enable(false)
    system_matrix_ptr = pointer_from_objref(dpsim_matrix)
    decomp_elapses = @belapsed decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, $system_matrix_ptr)) evals = samples

    solve_elapses = @belapsed solve(Base.unsafe_convert(Ptr{Cdouble}, $rhs_vector),
        Base.unsafe_convert(Ptr{Cdouble},
            zeros(Float64, length($rhs_vector)))) evals = samples
    GC.enable(true)

    # Restore the previous env
    ENV["JULIA_DEBUG"] = saved_debug_env

    BenchmarkResult(
        decomp_elapses,
        solve_elapses
    )
end

"""
    function benchmark(csr::SparseMatrixCSR, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))

Wrapper for `function benchmark(dpsim_matrix::dpsim_csr_matrix, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))`
"""
function benchmark(csr::SparseMatrixCSC, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))
    dpsim_matrix = Utils.csc_to_dpsim(csr)
    benchmark(dpsim_matrix, rhs_vector; samples=samples)
end

"""
    function benchmark(csr::SparseMatrixCSR, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))

Wrapper for `function benchmark(dpsim_matrix::dpsim_csr_matrix, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))`
"""
function benchmark(csr::SparseMatrixCSR, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))
    dpsim_matrix = Utils.csr_to_dpsim(csr)
    benchmark(dpsim_matrix, rhs_vector; samples=samples)
end

"""
    function benchmark(matrix::Matrix, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))

Wrapper for `function benchmark(dpsim_matrix::dpsim_csr_matrix, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))`
"""
function benchmark(matrix::Matrix, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))
    csr = Utils.to_zerobased_csr(matrix)
    benchmark(csr, rhs_vector; samples=samples)
end

"""
    function benchmark(matrix_path::AbstractString, rhs_path::AbstractString; samples::UInt=UInt(3))

Wrapper for `function benchmark(dpsim_matrix::dpsim_csr_matrix, rhs_vector::Vector{Float64}; samples::UInt=UInt(3))`
"""
function benchmark(matrix_path::AbstractString, rhs_path::AbstractString; samples::UInt=UInt(3))
    dpsim_matrix = Utils.read_input(Utils.ArrayPath(matrix_path))
    rhs_vector = Utils.read_input(Utils.VectorPath(rhs_path))
    benchmark(dpsim_matrix, rhs_vector; samples=samples)
end

"""
    function save_csv(path::AbstractString, benchmark_result::BenchmarkResult, matrix_path::String)

TODO: WORK IN PROGRESS
"""
function save_csv(path::AbstractString, benchmark_result::BenchmarkResult, strategy::Dict, matrix_path::String)
    data_frame = DataFrame(
        decomp_elapses=[benchmark_result.decomp_elapses],
        solve_elapses=[benchmark_result.solve_elapses],
        strategy=[strategy],
        matrix_path=[matrix_path]
    )

    # Create folder for csv
    mkpath(dirname("$path"))

    append = isfile("$path") # with append no header is written
    CSV.write("$path", data_frame; append=append)
end

end