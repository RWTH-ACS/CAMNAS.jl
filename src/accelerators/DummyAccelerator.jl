export DummyAccelerator, DummyAccelerator_LUdecomp
export discover_accelerator, mna_decomp, mna_solve

struct DummyAccelerator <: AbstractAccelerator
    name::String
    properties::AcceleratorProperties

    function DummyAccelerator(name::String = "dummy_accelerator", properties=AcceleratorProperties(true, 1, 1.0, 1.0))
        new(name, properties)
    end


end

struct DummyAccelerator_LUdecomp <: AbstractLUdecomp
    lu_decomp::LinearAlgebra.TransposeFactorization
end

function has_driver(accelerator::DummyAccelerator) 
    return true
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::DummyAccelerator) 
    
    try
        has_driver(accelerator)
    catch e
        @error "Dummy driver not found: $e"
        return
    end
    
    if !isempty(filter(x -> x.name == "dummy_accelerator", accelerators)) # check if cpu is already in accelerators_vector
        return
    end

    dummy_accelerator_perf = getPerformanceIndicator(accelerator)
    dummy_accelerator_power = get_tdp(accelerator)
    dummy_accelerator = DummyAccelerator("dummy_accelerator", AcceleratorProperties(true, 1, dummy_accelerator_perf, dummy_accelerator_power))
    push!(accelerators, dummy_accelerator)
end

# same implementation as NoAccelerator
function estimate_perf(accelerator::DummyAccelerator) # returns flops in GFLOPs
    
    return 400.0    #   choose an arbitrary performance value for dummyaccelerator

end

function get_tdp(accelerator::DummyAccelerator) # returns flops in GFLOPs
    
    return 400.0    #   choose an arbitrary powerconsumption value for dummyaccelerator

end
