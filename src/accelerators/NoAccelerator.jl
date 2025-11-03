export NoAccelerator, NoAccelerator_LUdecomp
export discover_accelerator, mna_decomp, mna_solve

struct NoAccelerator <: AbstractAccelerator 
    name::String
    properties::AcceleratorProperties
    

    function NoAccelerator(name::String = "cpu", properties=AcceleratorProperties(true, 1, 1.0, floatmax()))
        new(name, properties)
    end


end

struct NoAccelerator_LUdecomp <: AbstractLUdecomp 
    lu_decomp::LinearAlgebra.TransposeFactorization
end

function has_driver(accelerator::NoAccelerator)
    return true
end

function discover_accelerator(accelerators::Vector{AbstractAccelerator}, accelerator::NoAccelerator)
    
    try
        has_driver(accelerator)
    catch e
        @error "NoAccelerator driver not found: $e"
        return
    end
    
    if !isempty(filter(x -> x.name == "cpu", accelerators)) # check if cpu is already in accelerators_vector
        return
    end

    cpu_perf = getPerformanceIndicator(accelerator)
    cpu_power = get_tdp(accelerator)
    cpu = NoAccelerator("cpu", AcceleratorProperties(true, 1, cpu_perf, cpu_power))
    push!(accelerators, cpu)

end



function get_tdp(accelerator::NoAccelerator)
    return 95.0
end
