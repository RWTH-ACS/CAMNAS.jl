export MetalAccelerator, MetalAccelerator_LUdecomp
export discover_accelerator, mna_decomp, mna_solve

using Metal

struct MetalAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties


    function MetalAccelerator(name::String = "metal", properties=AcceleratorProperties(true, 1, 1.0, floatmax()))
        new(name, properties)
    end


end

# DenseMatrix since Sparse Matrix support for Metal.jl is not implemented yet.

struct MetalAccelerator_LUdecomp <: AbstractLUdecomp 
    lu_decomp::LU{Float32, MtlMatrix{Float32, Metal.PrivateStorage}, MtlVector{UInt32, Metal.PrivateStorage}}
    inverse::LU{Float32, MtlMatrix{Float32, Metal.PrivateStorage}, MtlVector{UInt32, Metal.PrivateStorage}}
end


function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::MetalAccelerator)
    @debug "discovering MetalAccelerator"

    if !isempty(filter(x -> x.name == "metal", accelerators)) # check if metal is already in accelerators_vector
        return
    end

    metal_perf = getPerformanceIndicator(accelerator)
    metal_power = get_tdp(accelerator)
    metal = MetalAccelerator("metal", AcceleratorProperties(true, 1, metal_perf, metal_power))
    push!(accelerators, metal)
    @debug "MetalAccelerator discovered and added to accelerators vector"
end

function has_driver(accelerator::MetalAccelerator)
    try
        !isempty(Metal.devices())
    catch e
        @error "Metal driver not found: $e"
        return false
    end
    return true
end


function estimate_perf(accelerator::MetalAccelerator;
                        n::Int = 4096, 
                        trials::Int = 5,
                        inT::DataType=Float32,
                        ouT::DataType=inT) # returns performance indicator


    A = Metal.mtl(ones(inT, n, n))
    B = Metal.mtl(ones(inT, n, n))
    C = Metal.mtl(zeros(ouT, n, n))

    
    
    min_time = @belapsed Metal.@sync mul!(C, A, B)


    flops = 2 * n^3 - n^2
    gflops = flops / (min_time * 1e9)

    perfIndicator = round(gflops, digits=2)

    return perfIndicator
end

function get_tdp(accelerator::MetalAccelerator)
    # TODO: Implement a proper TDP retrieval for Metal
    return 25.0
end


# Metal does not support ldiv or \, this is why we calculate with the inverse 
# either we calculate 2 lu decompositions or we move the calculated lu decomposition back to cpu, 
# to calculate the inverse only to then move the inverse back to the gpu, because there is no inverse function in Metal.jl

# this seems ineffective

function mna_decomp(sparse_mat, accelerator::MetalAccelerator)

    if rank(sparse_mat) < size(sparse_mat, 1)
        # matrix is not of full rank -> can convert, but not invertible
        @error "Matrix is not of full rank, cannot invert."
    end
    inverse = inv(sparse_mat)

    sparse_mat_mtl = Metal.mtl(sparse_mat)
    sparse_mat_mtl_inverse = Metal.mtl(inverse)

    decomp = lu(sparse_mat_mtl)
    inv_decomp = lu(sparse_mat_mtl_inverse) 
    
    return MetalAccelerator_LUdecomp(decomp, inv_decomp)
end

# Metal.jl does not include a solve function for LUdecomp yet and wrapper would be in objective-c
# so we use the inverse to solve the system
function mna_solve(system_matrix::MetalAccelerator_LUdecomp, rhs, accelerator::MetalAccelerator)
# FIXME:
# are dpsim matrices always nonsingular?

    # GPU solve step: not numerically stable, not as performant as \ or ldiv
    
    rhs_mtl = Metal.mtl(rhs)    
    lhs_mtl = Metal.mtl(system_matrix.inverse.P) * (system_matrix.inverse.L * (system_matrix.inverse.U * rhs_mtl))

    lhs = Array(lhs_mtl) # convert back to CPU array
    return Float64.(lhs) # convert back to Float64 

    # CPU solve step
    # lu_decomp = Matrix(system_matrix.lu_decomp)
    # return system_matrix.lu_decomp \ rhs

end
