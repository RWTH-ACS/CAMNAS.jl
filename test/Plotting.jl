module Plotting

using CSV
using DataFrames
using PlotlyJS

include("Utils.jl")

"""

"""
function plot_solve_vs_dimension(csv_path::String)
    # Read CSV
    df = CSV.read(csv_path, DataFrame)

    # CSV structure
    #decomp_elapses,solve_elapses,strategy(Dict),matrix_path

    # Convert column to Any so it can hold Dicts
    df.strategy = convert(Vector{Any}, df.strategy)

    # Convert all stringified Dicts into real Dicts
    df.strategy .= eval.(Meta.parse.(df.strategy))

    traces = Vector{PlotlyBase.AbstractTrace}()
    strategy_groups = groupby(df, :strategy)
    for strategy_group in strategy_groups
        solve_times = []
        dimensions = []
        for benchmark in eachrow(strategy_group)
            # Load Matrix
            matrix = Utils.read_input(Utils.ArrayPath(benchmark.matrix_path))
            push!(dimensions, matrix.row_number)

            push!(solve_times, benchmark.solve_elapses)

        end

        trace = scatter(
            x = dimensions,
            y = solve_times,
            name = strategy_group[1,:].strategy["specific_accelerator"]
        )

        push!(traces, trace)
        
    end

    layout = Layout(
        title = "Solve Time vs Matrix Dimension",
        xaxis_title = "Matrix Dimension",
        yaxis_title = "Solve Time",
        hovermode = "closest"
    )

    plt = plot(traces ,layout)
    display(plt)

    #return plt
end

begin
    plot_solve_vs_dimension("testBenchmark/test.csv")
end

end
