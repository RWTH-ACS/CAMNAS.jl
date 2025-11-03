module Accelerators
import ..CAMNAS

export AbstractAccelerator, AcceleratorProperties, AbstractLUdecomp
export discover_accelerator, mna_decomp, mna_solve, estimate_perf

using SparseArrays
using LinearAlgebra
using BenchmarkTools
# Hardwareawareness
abstract type AbstractAccelerator end

abstract type AbstractLUdecomp end

struct AcceleratorProperties
    availability::Bool
    priority::Int64
    performanceIndicator::Float64      
    power_watts::Float64            # max Power usage
    energy_efficiency::Float64      # performanceIndicator/W

    function AcceleratorProperties(availability::Bool, priority::Int64, performanceIndicator::Float64, power_watts::Float64) 
        new(availability, priority, performanceIndicator, power_watts, round(performanceIndicator/power_watts, digits=4))
    end

    function AcceleratorProperties()
        new(true, 1, 1.0, 1.0, 1.0)
    end

end


# not in function on purpose, otherwise scope issue with include statements

# include all accelerator files
global accelerator_files
accelerator_files = Vector()
read(`pwd`, String)
for file in readdir(dirname(@__FILE__), join=true)
    if endswith(file, ".jl") && basename(file) != "Accelerators.jl" && basename(file) ∉ accelerator_files
        @debug "file found $file"
        push!(accelerator_files, basename(file))
        include(basename(file))
    end
end
@debug accelerator_files





function load_all_accelerators(accelerators::Vector{AbstractAccelerator})   # Accelerator structs are called like the .jl file
    global accelerator_files
    if isempty(accelerators)
        for file in accelerator_files
            structname = split(file, ".")[1]
            symbol =  Symbol(structname)

            try
                if !isdefined(Accelerators, symbol)
                    @warn "No struct named '$structname' found in module Accelerators."
                    continue
                end

                accelerator_type = getfield(Accelerators, symbol)

                if !isdefined(@__MODULE__, :discover_accelerator)
                    @warn "No function `discover_accelerator` defined for file '$file'."
                    continue
                end

                    instance = accelerator_type()
        
                    if !has_driver(instance)
                        @error "Driver not present for $structname."
                        continue
                    end
        
                    discover_accelerator(accelerators, instance)
            catch e
                @error "Error loading accelerator from file '$file': $e"
            end

            
        end
    end
end

function has_driver(accelerator::AbstractAccelerator)
    @error "has_driver not implemented for $(typeof(accelerator))"
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::AbstractAccelerator)

    @error "discover_accelerator not implemented for $(typeof(accelerator))"

end


function mna_decomp(sparse_mat, accelerator::AbstractAccelerator)
    lu_decomp = SparseArrays.lu(sparse_mat) |> NoAccelerator_LUdecomp
    return lu_decomp
end


function mna_solve(system_matrix::AbstractLUdecomp, rhs, accelerator::AbstractAccelerator)
    return system_matrix.lu_decomp \ rhs
end

function estimate_perf(accelerator::AbstractAccelerator; 
                        n::Int = 8192, 
                        trials::Int = 5,
                        inT::DataType=Float64,
                        ouT::DataType=inT)

    A = ones(inT, n, n)
    B = ones(inT, n, n)
    C = zeros(ouT, n, n)

    # TODO: How can we create big random matrices where LU decomp exists?

    # idea: do not benchmark matrix multiplication, but LU decomp and solve step and calculate performance indicator from that

    min_time = @belapsed mul!($C, $A, $B)

    flops = 2 * n^3 - n^2
    perfIndicator = round(flops / (min_time * 1e9), digits=2) 

    return perfIndicator

end

function get_tdp(accelerator::AbstractAccelerator)
    return floatmax(Float64)
end


"""
Set the accelerator device for the given accelerator.
This function needs to be implemented in case there are multiple accelerator devices of the same type.
"""
function set_acceleratordevice!(accelerator::AbstractAccelerator)
    #@warn "set_acceleratordevice not implemented or necessary for $(typeof(accelerator))"
end

# Get the FLOPs of the accelerator from a file or benchmark and save to file.
function getPerformanceIndicator(accelerator::AbstractAccelerator)
    filepath = "$(@__DIR__)/.acceleratorPerformanceIndicator"
    perf = nothing

    @debug "Checking for performance in file: $filepath and file exists: $(isfile(filepath))"
    

    if isfile(filepath) # check if the file exists and try to read FLOPs
        open(filepath, "r") do file
            for line in eachline(file)
                if occursin(accelerator.name, line)
                    parts = split(line)
                    if length(parts) >= 2
                        perf = parse(Float64, parts[end]) 
                        @debug "Performance Infication found for $(accelerator.name): $perf, skip benchmarking"
                        break
                    end
                end
            end
        end
        if perf === nothing   # Performance Infication not found for this accelerator
            perf = estimate_perf(accelerator)
            open(filepath, "a") do file
                println(file, "$(accelerator.name) $perf")
            end
        end
    else
        perf = estimate_perf(accelerator)
        open(filepath, "w") do file
            println(file, "$(accelerator.name) $perf")  # create the file with initial value
        end
    end



    return perf

end



end