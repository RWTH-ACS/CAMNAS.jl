export CUDAccelerator, CUDAccelerator_LUdecomp
export discover_accelerator, mna_decomp, mna_solve

using CUDA
using CUDA.CUSPARSE
using CUSOLVERRF
using SparseMatricesCSR

"""
    CUDAccelerator <: AbstractAccelerator

Concrete accelerator type representing an NVIDIA CUDA-capable GPU device for CAMNAS.

This struct wraps a CUDA device and its associated properties (performance, power, etc.) for use
by CAMNAS.jl accelerator selection logic.

# Fields
- `name::String` : human-readable device name (e.g., "NVIDIA GeForce RTX 3090").
- `properties::AcceleratorProperties` : measured or estimated performance and power characteristics.
- `device::CuDevice` : the underlying CUDA device handle.
"""
struct CUDAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties
    device::CuDevice

    function CUDAccelerator(name::String = "cuda", dev::CuDevice = CUDA.device() , properties=AcceleratorProperties(true, 1, 1.0, floatmax()))
        new(name, properties, dev)
    end
end

"""
    CUDAccelerator_LUdecomp <: AbstractLUdecomp

Wrapper for a GPU LU factorization computed via CUSOLVERRF for sparse matrices.

This struct encapsulates a `CUSOLVERRF.RFLU` object, which holds the refactorized LU decomposition
on the GPU.

# Fields
- `lu_decomp::CUSOLVERRF.RFLU` : the GPU-resident LU factorization object.
"""
struct CUDAccelerator_LUdecomp <: AbstractLUdecomp 
    lu_decomp::CUSOLVERRF.RFLU
end

function has_driver(accelerator::CUDAccelerator)
    try
        CUDA.has_cuda()
    catch e
        @warn "CUDA driver not found: $e"
        return false
    end
    return true
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::CUDAccelerator) 
    devices = collect(CUDA.devices())   # Vector of CUDA devices 
    @debug "Found $(length(devices)) CUDA devices"

    for device in devices
        device_name = CUDA.name(device)*"($(device.handle))"
        cuda_acc = CUDAccelerator(device_name, device)
        power_limit = get_tdp(cuda_acc)
        cuda_perf = getPerformanceIndicator(cuda_acc)
        cuda_acc = CUDAccelerator(device_name, device, AcceleratorProperties(true, 1, cuda_perf, power_limit))
        push!(accelerators, cuda_acc)
    end
    
end

function mna_decomp(sparse_mat, accelerator::CUDAccelerator)
    @debug "Calculate Decomposition on $(CUDA.device()) on Thread $(Threads.threadid())"
    @debug "Calculating on $(accelerator.name)"
    matrix = CuSparseMatrixCSR(CuArray(sparse_mat)) # Sparse GPU implementation
    lu_decomp = CUSOLVERRF.RFLU(matrix; symbolic=:RF) |> CUDAccelerator_LUdecomp

    return lu_decomp
end

function mna_solve(system_matrix::CUDAccelerator_LUdecomp, rhs, accelerator::CUDAccelerator)
    @debug "Calculate Solve step on $(CUDA.device())"
    rhs_d = CuVector(rhs)
    ldiv!(system_matrix.lu_decomp, rhs_d)
    return Array(rhs_d)
end

function estimate_perf(accelerator::CUDAccelerator;
                        n::Int = 8192, 
                        trials::Int = 5,
                        inT::DataType=Float64,
                        ouT::DataType=inT)   # returns flops in GFLOPs
    dev::CUDA.CuDevice = accelerator.device
    @debug "Estimating performance Indication for CUDA device $(dev.handle) with benchmarking"

    # Set the CUDA device for benchmark
    CUDA.device!(dev)
    
    # Allocate GPU matrices
    A = CUDA.ones(inT, n, n)
    B = CUDA.ones(inT, n, n)
    C = CUDA.zeros(ouT, n, n)

    flops = 2 * n^3 - n^2

    min_time = @belapsed CUDA.@sync mul!($C, $A, $B) 

    gflops = flops / (min_time * 1e9)

    perfIndicator = round(gflops, digits=2)
  
    return perfIndicator
end


function get_tdp(accelerator::CUDAccelerator)
    mapping = map_CuDevice_to_nvidiasmi()
    cuda_device_id = accelerator.device.handle

    cmd = `nvidia-smi -i $(mapping[cuda_device_id]) --query-gpu=power.limit --format=csv,noheader,nounits`
    power_limit = readlines(cmd)
    power_limit = parse(Float64, power_limit[1]) 
    return power_limit
end


function set_acceleratordevice!(accelerator::CUDAccelerator)
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
    
    idx = findfirst(x-> typeof(x) == CUDAccelerator_LUdecomp, CAMNAS.system_matrix)
    cuda_lu = system_matrix_dev2host(CAMNAS.system_matrix[idx])

    # Switch to new CUDA device
    CUDA.device!(accelerator.device)
    @debug "Current CUDA device is now $(CUDA.device())"

    # Recreate LU decompositions on the new device
    CAMNAS.system_matrix[idx] = mna_decomp(cuda_lu, accelerator)
    @debug "Successfully migrated LU decomposition to device $(accelerator.device)"
end

function map_CuDevice_to_nvidiasmi()
    # collect CuDevice PCI bus IDs
    cuda_devices = Dict{Int, String}()
    for i in 0:length(CUDA.devices()) - 1
        dev = CuDevice(i)
        pci = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_PCI_BUS_ID)
        pci_hex = string(pci, base=16) |> uppercase
        cuda_devices[i] = pci_hex
    end

    # collect nvidia-smi device list with PCI and IDs
    smi_output = readlines(`nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader`)
    smi_devices = Dict{String, Int}()
    for line in smi_output
        idx, pci_full = split(strip(line), ',')
        idx = parse(Int, strip(idx))
        pci_bus_id = strip(pci_full)[10:11]  # extrace "XX" from "00000000:XX:00.0"
        smi_devices[pci_bus_id] = idx
    end

    mapping = Dict{Int, Int}()
    for (i, pci) in cuda_devices
        if haskey(smi_devices, pci)
            mapping[i] = smi_devices[pci]
        else
            @warn "No matching nvidia-smi device found for CuDevice($i) (PCI $pci)"
        end
    end

    return mapping
end

function system_matrix_dev2host(cuda_lu::CUDAccelerator_LUdecomp) #transfer LU factorization from CUSOLVERRF.RFLU to SparseArrays.UMFPACK.UMFPACKLU type
    # Access combined LU matrix (GPU, CSR format)
    M_gpu = cuda_lu.lu_decomp.M

    rowPtr = collect(M_gpu.rowPtr)
    colVal = collect(M_gpu.colVal)
    nzVal = collect(M_gpu.nzVal)
    nrow = size(M_gpu, 1)
    ncol = size(M_gpu, 2)

    M_cpu = SparseMatrixCSR{1}(nrow, ncol, rowPtr, colVal, nzVal) # 1 indicates index base
    return  M_cpu
end
