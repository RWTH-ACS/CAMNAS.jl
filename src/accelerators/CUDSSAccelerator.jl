export CUDSSAccelerator, CUDSSAccelerator_LUdecomp
export discover_accelerator, mna_decomp, mna_solve

using CUDSS

using CUDA
using CUDA.CUSPARSE
using CUSOLVERRF
using SparseMatricesCSR

"""
    CUDSSAccelerator <: AbstractAccelerator

Concrete accelerator type representing an NVIDIA CUDA-capable GPU device for CAMNAS.

This struct wraps a CUDA device and its associated properties (performance, power, etc.) for use
by CAMNAS.jl accelerator selection logic.

# Fields
- `name::String` : human-readable device name (e.g., "NVIDIA GeForce RTX 3090").
- `properties::AcceleratorProperties` : measured or estimated performance and power characteristics.
- `device::CuDevice` : the underlying CUDA device handle.
"""
struct CUDSSAccelerator <: AbstractAccelerator
    name::String
    properties::AcceleratorProperties
    device::CuDevice

    function CUDSSAccelerator(name::String="cuddss", dev::CuDevice=CUDA.device(), properties=AcceleratorProperties(true, 1, 1.0, floatmax()))
        new(name, properties, dev)
    end
end

"""
    CUDSSAccelerator_LUdecomp <: AbstractLUdecomp

Wrapper for a GPU LU factorization computed via CUSOLVERRF for sparse matrices.

This struct encapsulates a `CUSOLVERRF.RFLU` object, which holds the refactorized LU decomposition
on the GPU.

# Fields
- `lu_decomp::CUSOLVERRF.RFLU` : the GPU-resident LU factorization object.
"""
struct CUDSSAccelerator_LUdecomp <: AbstractLUdecomp
    solver::CUDSS.CudssSolver
    lu_decomp

    function CUDSSAccelerator_LUdecomp(solver::CUDSS.CudssSolver)
        new(solver, zeros(solver.matrix.nrows, solver.matrix.nrows))
    end
end
# size(decomp:: CUDSSAccelerator_LUdecomp) = [decomp.matrix.nrows, decomp.matrix.nrows]
# CUDSSAccelerator_LUdecomp(solver::CUDSS.CudssSolver) = CUDSSAccelerator_LUdecomp(solver, [solver.matrix.nrows])

function has_driver(accelerator::CUDSSAccelerator)
    try
        CUDA.has_cuda()
    catch e
        @warn "CUDA driver not found: $e"
        return false
    end
    return true
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::CUDSSAccelerator)
    devices = collect(CUDA.devices())   # Vector of CUDA devices
    @debug "Found $(length(devices)) CUDA devices"

    for device in devices
        device_name = "CUDSS "*CUDA.name(device)*"($(device.handle))"
        cuda_acc = CUDSSAccelerator(device_name, device)
        power_limit = get_tdp(cuda_acc)
        cuda_perf = getPerformanceIndicator(cuda_acc)
        cuda_acc = CUDSSAccelerator(device_name, device, AcceleratorProperties(true, 1, cuda_perf, power_limit))
        push!(accelerators, cuda_acc)
    end

end

function mna_decomp(sparse_mat, accelerator::CUDSSAccelerator)
    @debug "Calculate Decomposition on $(CUDA.device()) on Thread $(Threads.threadid())"
    @debug "Calculating on $(accelerator.name)"

    n = size(sparse_mat, 1)
    x_cpu = zeros(Float64, n)
    b_cpu = rand(Float64, n)

    a_gpu = CuSparseMatrixCSR(CuArray(sparse_mat)) # Sparse GPU implementation
    solver = CudssSolver(a_gpu, "G", 'F')
    x_gpu = CuVector(x_cpu)
    b_gpu = CuVector(b_cpu)

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)

    lu_wrapper = solver |> CUDSSAccelerator_LUdecomp

    return lu_wrapper
end

function mna_solve(lu_wrapper::CUDSSAccelerator_LUdecomp, rhs, accelerator::CUDSSAccelerator)
    @debug "Calculate Solve step with CUDSS on $(CUDA.device())"
    b_gpu = CuVector(rhs)

    x_cpu = zeros(Float64, lu_wrapper.solver.matrix.nrows)
    x_gpu = CuVector(x_cpu)

    # TODO: Verify solver returned from wrapper is ok
    cudss("solve", lu_wrapper.solver, x_gpu, b_gpu)

    @debug x_gpu

    return Array(x_gpu)
end

function set_acceleratordevice!(accelerator::CUDSSAccelerator)
    # This function is used to set the CUDA device for the current thread
    # It is called by the CAMNAS.jl module to ensure that the correct device is used
    if accelerator.device == CUDA.device()
        @debug "CUDA device $(accelerator.device) is already set on Thread $(Threads.threadid())"
        return
    end

    old_device = CUDA.device()
    @debug "Setting CUDA device to $(accelerator.device) on Thread $(Threads.threadid())"
    @debug "Previous device was $(old_device)"
    @debug "Extracting LU decomposition from device $(old_device)"

    idx = findfirst(x->typeof(x) == CUDSSAccelerator_LUdecomp, CAMNAS.system_matrix)
    cuda_lu = system_matrix_dev2host(CAMNAS.system_matrix[idx])

    # Switch to new CUDA device
    CUDA.device!(accelerator.device)
    @debug "Current CUDA device is now $(CUDA.device())"

    # Recreate LU decompositions on the new device
    CAMNAS.system_matrix[idx] = mna_decomp(cuda_lu, accelerator)
    @debug "Successfully migrated LU decomposition to device $(accelerator.device)"
end
