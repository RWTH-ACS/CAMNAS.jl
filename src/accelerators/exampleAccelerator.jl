export exampleAccelerator, exampleAccelerator_LUdecomp
export discover_accelerator, mna_decomp, mna_solve


""" Create Struct for new Accelerator. 
    Struct needs to be a subtype of `AbstractAccelerator` and Structname has to be the same as file name.
    with `name` and `properties` fields.
    Create two constructors: one default and one with passing parameters.
"""

struct exampleAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties


    function exampleAccelerator(name::String = "example", properties=AcceleratorProperties(true, 1, 1.0, floatmax()))
        new(name, properties)
    end

    

end


""" Create Struct for LU decomposition of the exampleAccelerator.
    Struct needs to be a subtype of `AbstractLUdecomp` and Structname has to be the same as file name with `_LUdecomp` suffix.
    with `lu_decomp` field.
"""

struct exampleAccelerator_LUdecomp <: AbstractLUdecomp
    lu_decomp::LinearAlgebra.TransposeFactorization
end

""" discover_accelerator function
Check if accelerator is already in global `accelerators` vector. 
Call `estimate_perf` and `get_tdp` function.
Create new exampleAccelerator struct object and push it into the `accelerators` vector.

If there are multiple accelerator of the same type, make sure to discover all of them.
"""

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::exampleAccelerator)
    if !isempty(filter(x -> x.name == "example", accelerators)) 
        return
    end

    example_perf = getPerformanceIndicator(accelerator)
    example_power = get_tdp(accelerator)
    example = exampleAccelerator("example", AcceleratorProperties(true, 1, example_perf, example_power))
    push!(accelerators, example)

end

""" has_driver function
Check if the driver for the exampleAccelerator is available.
Either use a specific function from the accelerator package or check if device list is not empty.
"""

function has_driver(accelerator::exampleAccelerator)
    return false 
end

""" estimate_perffunction
Estimate the performance Indicator of the exampleAccelerator by running a simple matrix multiplication benchmark.

Overload the function from AbstractAccelerator and adjust the Matrices A, B and C to the accelerator specific type of matrix 
    (e.g. MtlMatrix for Metal or CuArray for CUDA).
Check for capabilities to run FP32 or FP64
"""

function estimate_perf(accelerator::exampleAccelerator;
                        n::Int = 4096, 
                        trials::Int = 5,
                        inT::DataType=Float64,
                        ouT::DataType=inT)

end

""" mna_decomp function
Perform LU decomposition on the system matrix using the exampleAccelerator.
Calculate the LU decomposition using the Accelerator's specific method.
"""
function mna_decomp(sparse_mat, accelerator::exampleAccelerator)
    lu_decomp = SparseArrays.lu(sparse_mat) |> exampleAccelerator_LUdecomp
    return lu_decomp
end

""" mna_solve function
Solve system matrix with the right-hand side vector using the exampleAccelerator.
Use `\` or `ldiv!` to solve the linear system.

Make sure to convert the system_matrix to the appropriate type if necessary.
DPsim expects a Float64 vector for the solution.
"""

function mna_solve(system_matrix::AbstractLUdecomp, rhs, accelerator::exampleAccelerator)
    return system_matrix.lu_decomp \ rhs
end

""" get_tdp function
Return the Thermal Design Power (TDP) of the exampleAccelerator.
Relevant for power strategy and performance estimation.
Not easy to get this value from system.
"""
function get_tdp(accelerator::exampleAccelerator)
    return floatmax(Float64)
end