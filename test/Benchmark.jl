#=
Author: Pascal Bauer <pascal.bauer@rwth-aachen.de>
SPDX-FileCopyrightText: 2025 Pascal Bauer <pascal.bauer@rwth-aachen.de>
=#

module Benchmark

using CAMNAS

using Base
using BenchmarkTools
using CSV, DataFrames

function get_next_folder(base_dir="benchmark/")
    runs = filter(name -> occursin(r"^run_\d+$", name), readdir(base_dir))

    if isempty(runs)
        return joinpath(base_dir, "run_1")
    end

    nums = parse.(Int, replace.(runs, r"^run_" => ""))
    next_num = maximum(nums) + 1

    return joinpath(base_dir, "run_$(next_num)")
end

function benchmark_solve(rhs::Vector{Float64})
    solve = @elapsed begin
        lhs = zeros(Float64, length(rhs))
        CAMNAS.solve(Base.unsafe_convert(Ptr{Cdouble}, rhs), Base.unsafe_convert(Ptr{Cdouble}, lhs))
    end

    Dict("solve" => solve)
end

"""
    function save_csv(path::AbstractString, benchmark_metrics::Benchmarkmetrics, matrix_path::String)

"""
function save_csv(path::AbstractString, metrics::Dict, strategy::Dict, matrix_path::String)
    data_frame = DataFrame(
        metrics=[metrics],
        strategy=[strategy],
        matrix_path=[matrix_path]
    )

    # Create folder for csv
    mkpath(dirname("$path"))

    append = isfile("$path") # with append no header is written
    CSV.write("$path", data_frame; append=append)
end

end