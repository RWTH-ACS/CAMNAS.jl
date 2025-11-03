export CUDAccelerator, CUDAccelerator_LUdecomp
export discover_accelerator, mna_decomp, mna_solve

using CUDA
using CUDA.CUSPARSE
using CUSOLVERRF

using Printf

struct CUDAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties
    device::CuDevice


    function CUDAccelerator(name::String = "cuda", dev::CuDevice = CUDA.device() , properties=AcceleratorProperties(true, 1, 1.0, floatmax()))
        new(name, properties, dev)
    end


end

struct CUDAccelerator_LUdecomp <: AbstractLUdecomp 
    lu_decomp::CUSOLVERRF.RFLU
end

function has_driver(accelerator::CUDAccelerator)
    try
        CUDA.has_cuda()
    catch e
        @error "CUDA driver not found: $e"
        return false
    end
    return true
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::CUDAccelerator) 

    try
        has_driver(accelerator)
    catch e
        @error "CUDA driver not found: $e"
        return
    end

    devices = collect(CUDA.devices())   # Vector of CUDA devices 
    @debug "Found $(length(devices)) CUDA devices"

    for dev in devices 
        cuda_acc = CUDAccelerator(CUDA.name(dev), dev)
        power_limit = get_tdp(cuda_acc)
        cuda_perf = getPerformanceIndicator(cuda_acc)
        cuda_acc = CUDAccelerator(CUDA.name(dev), dev, AcceleratorProperties(true, 1, cuda_perf, power_limit))
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


function set_acceleratordevice!(acc::CUDAccelerator)
    # This function is used to set the CUDA device for the current thread
    # It is called by the CAMNAS.jl module to ensure that the correct device is used

    if acc.device == CUDA.device()
        @debug "CUDA device $(acc.device) is already set on Thread $(Threads.threadid())"
        return
    end


    @debug "Setting CUDA device to $(acc.device) on Thread $(Threads.threadid())"
    CUDA.device!(acc.device)
    @debug "Current CUDA device is now $(CUDA.device())"
end

function map_CuDevice_to_nvidiasmi()
    # collect CuDevice PCI bus IDs
    cuda_devices = Dict{Int, String}()
    for i in 0:length(CUDA.devices()) - 1
        dev = CuDevice(i)
        pci = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_PCI_BUS_ID)
        pci_hex = @sprintf("%02x" , pci) |> uppercase
        cuda_devices[i] = pci_hex
    end

    # collect nvidia-smi device list with PCI and IDs
    smi_output = readlines(`nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader`)
    smi_devices = Dict{String, Int}()
    for line in smi_output
        #isempty(strip(line)) && continue
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