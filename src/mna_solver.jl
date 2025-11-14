using SparseArrays
using LinearAlgebra
using FileWatching

include("accelerators/Accelerators.jl")
include("config.jl")

using .Accelerators

# Accelerator Selection
abstract type AbstractSelectionStrategy end

struct DefaultStrategy <: AbstractSelectionStrategy end #choose first accelerator from a defined order
struct LowestPowerStrategy <: AbstractSelectionStrategy end
struct HighestPerfStrategy <: AbstractSelectionStrategy end
struct NoStrategy <: AbstractSelectionStrategy end
struct SpecificAcceleratorStrategy <: AbstractSelectionStrategy end

# Vector of available accelerators
global accelerators_vector = Vector{AbstractAccelerator}()
global system_environment = Channel(1)
global current_accelerator = NoAccelerator()
global current_strategy = NoStrategy()
global system_matrix = Vector{AbstractLUdecomp}()

"""
    determine_accelerator(strategy::AbstractSelectionStrategy, accelerators_vector::Vector{AbstractAccelerator}) -> Nothing

Falling back to DefaultStrategy if no specific strategy is implemented.
"""
function determine_accelerator(strategy::AbstractSelectionStrategy, accelerators_vector::Vector{AbstractAccelerator})
    @debug "Strategy not implemented, falling back to DefaultStrategy"
    determine_accelerator(DefaultStrategy(), accelerators_vector)
end

"""
    determine_accelerator(strategy::DefaultStrategy, accelerators_vector::Vector{AbstractAccelerator}) -> Nothing

The default strategy implements a GPU first approach.
It selects the first available GPU-based accelerator, if none are available it falls back to the CPU.
"""
function determine_accelerator(strategy::DefaultStrategy, accelerators_vector::Vector{AbstractAccelerator})
    allowed_accelerators = Vector{AbstractAccelerator}()

    # This really doesnt seem nice
    if varDict["allow_gpu"]
        allowed_accelerators = filter(x -> typeof(x) != NoAccelerator, accelerators_vector)
    end

    if varDict["allow_cpu"]
        push!(allowed_accelerators, accelerators_vector[findfirst(x -> typeof(x) == NoAccelerator, accelerators_vector)])
    end

    if isempty(allowed_accelerators)
        @error "No accelerators available for selection."
        return nothing
    end

    idx = 1 # choose first accelerator from the available accelerators_vector
    set_current_accelerator!(allowed_accelerators[idx])
    @debug "DefaultStrategy selected, using $(accelerators_vector[idx])"
end

"""
    determine_accelerator(strategy::NoStrategy, accelerators_vector::Vector{AbstractAccelerator}) -> Nothing

The NoStrategy does not change the current accelerator.
"""
function determine_accelerator(strategy::NoStrategy, accelerators_vector::Vector{AbstractAccelerator})
    @debug "NoStrategy selected"
end

"""
    determine_accelerator(strategy::LowestPowerStrategy, accelerators_vector::Vector{AbstractAccelerator}) -> Nothing

The LowestPowerStrategy selects the accelerator with the lowest power consumption property, defined in AcceleratorProperties.
"""
function determine_accelerator(strategy::LowestPowerStrategy, accelerators_vector::Vector{AbstractAccelerator})
    allowed_accelerators = Vector{AbstractAccelerator}()

    if !varDict["allow_gpu"] && !varDict["allow_cpu"]
        @error "Nothing allowed"
        return nothing
    end 

    # This really doesnt seem nice
    if varDict["allow_gpu"]
        allowed_accelerators = filter(x -> typeof(x) != NoAccelerator, accelerators_vector)
    end

    if varDict["allow_cpu"]
        push!(allowed_accelerators, accelerators_vector[findfirst(x -> typeof(x) == NoAccelerator, accelerators_vector)])
    end

    if isempty(allowed_accelerators)
        @error "No accelerators available for selection."
        return nothing
    end

    value, index = findmin(x -> x.properties.power_watts, allowed_accelerators)
    set_current_accelerator!(allowed_accelerators[index])
end

"""
    determine_accelerator(strategy::HighestPerfStrategy, accelerators_vector::Vector{AbstractAccelerator}) -> Nothing

The HighestPerfStrategy selects the accelerator with the highest performance indicator property, defined in AcceleratorProperties.
"""
function determine_accelerator(strategy::HighestPerfStrategy, accelerators_vector::Vector{AbstractAccelerator})
    allowed_accelerators = Vector{AbstractAccelerator}()

    if !varDict["allow_gpu"] && !varDict["allow_cpu"]
        @error "Nothing allowed"
        return nothing
    end 

    # This really doesnt seem nice
    if varDict["allow_gpu"]
        allowed_accelerators = filter(x -> typeof(x) != NoAccelerator, accelerators_vector)
    end
    
    if varDict["allow_cpu"]
        push!(allowed_accelerators, accelerators_vector[findfirst(x -> typeof(x) == NoAccelerator, accelerators_vector)])
    end

    if isempty(allowed_accelerators)
        @error "No accelerators available for selection."
        return nothing
    end

    value, index = findmax(x -> x.properties.performanceIndicator, allowed_accelerators)
    set_current_accelerator!(allowed_accelerators[index])
end

function determine_accelerator(strategy::SpecificAcceleratorStrategy, accelerators_vector::Vector{AbstractAccelerator})
    specific_acc_name = varDict["specific_accelerator"]

    if specific_acc_name === nothing
        @error "No specific accelerator name provided in 'specific_accelerator' variable."
        return nothing
    end

    idx = findfirst(x -> x.name == specific_acc_name, accelerators_vector)
    if idx === nothing
        @error "Specified accelerator '$specific_acc_name' not found among available accelerators."
        return nothing
    end

    set_current_accelerator!(accelerators_vector[idx])
    @debug "SpecificAcceleratorStrategy selected, using $(accelerators_vector[idx])"
end


function find_accelerator()
    global accelerators_vector
    try
        @debug "Trying to load accelerators..."
        Accelerators.load_all_accelerators(accelerators_vector)
    catch e 
        @error "Failed to load accelerators: $e"
        set_current_accelerator!(NoAccelerator()) 
    end
    
    evaluate_system_environment(nothing)
    @debug "Present accelerators: $([a.name for a in accelerators_vector])"
end

function systemcheck()
    if varDict["hwAwarenessDisabled"]
        @info "[CAMNAS] Hardware awareness disabled... Using Fallback implementation"
        set_current_accelerator!(NoAccelerator())
    else
        find_accelerator()
    end
end

function file_watcher()
    @debug "File watcher is running on Thread $(Threads.threadid())"
    file_system_env = (@__DIR__)*"/system.env"
    @debug "Watching sytem environment at : $file_system_env"
    global run
    while run
        # @debug "Waiting for file change..."
        fw = watch_file(file_system_env, 3)
        if fw.changed
            @debug "Filewatcher triggered!"
            content = read(file_system_env, String)
            if isready(system_environment)
                take!(system_environment)
            end

            put!(system_environment, content)
            @debug "System environment updated!"
        end
    end
    @debug "File watcher stopped!"
end

function determine_accelerator()
    global accelerators_vector
    @debug "determine_accelerator is running on Thread $(Threads.threadid())"
    while true
        val = take!(system_environment)
        @debug "Received new system environment!: $val"
        val === nothing ? break : nothing
        
        evaluate_system_environment(val)
    end
    @debug "Accelerator determination stopped!"
end

function evaluate_system_environment(content)
    global current_strategy
    first_run = false
    if(content === nothing)
        @debug "Setting up: Reading ENV for the first time"
        file_system_env = (@__DIR__)*"/system.env"
        first_run = true
        content = read(file_system_env, String)
    end

    for line in split(content, '\n')[2:end]
        if length(line) == 0
            continue
        end
        key, value = split(line; limit=2)
        try
            varDict[key] = parse(Bool, value)
        catch ArgumentError
            varDict[key] = value
        end
    end

    @debug "Allow CPU is: $(varDict["allow_cpu"])"
    @debug "Allow GPU is: $(varDict["allow_gpu"])"
    @debug "$varDict"

    # Currently, force statments are the strongest, then consider strategies
    if varDict["runtime_switch"] || first_run

        # FORCING
        if varDict["force_cpu"] || varDict["force_gpu"]
            if varDict["force_cpu"] && varDict["force_gpu"]
                @warn "Conflict: Both 'force_cpu' and 'force_gpu' are set. Only one can be forced."
                idx = findfirst(x -> x.name == "cpu", accelerators_vector)
                typeof(current_accelerator) == NoAccelerator || set_current_accelerator!(accelerators_vector[idx])
            
            elseif varDict["allow_gpu"] && varDict["force_gpu"] # anything but cpu is considered gpu
                idx = findfirst(x -> typeof(x) != NoAccelerator, accelerators_vector)   
                set_current_accelerator!(accelerators_vector[idx])
            
            elseif varDict["allow_cpu"] && varDict["force_cpu"]
                idx = findfirst(x -> x.name == "cpu", accelerators_vector)
                typeof(current_accelerator) == NoAccelerator || set_current_accelerator!(accelerators_vector[idx])
            end
            @debug "Forcing prioritized, using NoStrategy"
            determine_accelerator(NoStrategy(), accelerators_vector)

        # STRATEGIES
        elseif varDict["allow_strategies"]
            if varDict["highest_flop_strategy"] && varDict["lowest_power_strategy"]
                @debug "Too many Stragegies set! Only one can be used at a time."
                current_strategy = DefaultStrategy()

            elseif varDict["highest_flop_strategy"]
                @debug "Selected HighestPerfStrategy"
                current_strategy = HighestPerfStrategy()
            
            elseif varDict["lowest_power_strategy"] 
                @debug "Selected LowestPowerStrategy"
                current_strategy = LowestPowerStrategy()

            elseif varDict["specific_accelerator_strategy"]
               @debug "Selected SpecificAcceleratorStrategy"
               current_strategy = SpecificAcceleratorStrategy()

            else
                @debug "Selected DefaultStrategy"
                current_strategy = DefaultStrategy()
            end
            determine_accelerator(current_strategy, accelerators_vector)

        elseif varDict["allow_gpu"] 
            idx = findfirst(x -> typeof(x) != NoAccelerator, accelerators_vector)   
            set_current_accelerator!(accelerators_vector[idx])
        
        elseif varDict["allow_cpu"]
            idx = findfirst(x -> x.name == "cpu", accelerators_vector)
            typeof(current_accelerator) == NoAccelerator || set_current_accelerator!(accelerators_vector[idx])
        
        # NOTHING ALLOWED
        else
            @debug "Conflict: Nothing is allowed. THIS DOESNT MAKE SENSE!"
        end

        if varDict["allow_strategies"]
            @info "[CAMNAS] Currently used strategy: $(typeof(current_strategy))"
        end
        @info "[CAMNAS] Currently used accelerator: $current_accelerator" 
    else
        @warn "Runtime switch is disabled, Accelerator will not be changed."
    end
end    

# FIXME: this seems weird not to be in Accelerators.jl
function set_current_accelerator!(acc::AbstractAccelerator) 
    @debug "Setting current accelerator to: $(typeof(acc))"
    global current_accelerator = acc
end

# Housekeeping
function mna_init(sparse_mat)
    global varDict = parse_env_vars()
    create_env_file()

    systemcheck()
    global run = true
    global csr_mat = sparse_mat

    global fw = Threads.@spawn file_watcher()
    global da = Threads.@spawn determine_accelerator()
end

function mna_cleanup()
    global run = false

    wait(fw)
    put!(system_environment, nothing) # Signal to stop
    wait(da)
    close(system_environment)
    @debug "Cleanup done!"
end

# Solving Logic
function set_csr_mat(csr_matrix)
    global csr_mat = csr_matrix
end

"""
    get_ludecomp_type(accelerator::AbstractAccelerator) -> Type

Given an accelerator object, returns the corresponding LU decomposition type
by naming convention: <AcceleratorType>_LUdecomp
"""
function get_ludecomp_type(accelerator::AbstractAccelerator)
    acc_type = typeof(accelerator)
    acc_name = string(nameof(typeof(accelerator)))  
    lu_type_name = Symbol(acc_name * "_LUdecomp")  

    if !isdefined(Accelerators, lu_type_name)
        error("LU decomposition type $lu_type_name not defined in module $(mod).")
    end

    return getfield(Accelerators, lu_type_name)
end

"""
    mna_decomp(sparse_mat::SparseMatrixCSC) -> Vector{AbstractLUdecomp}

Performs the MNA decomposition of the given sparse matrix using the current accelerator.
"""
function mna_decomp(sparse_mat)
    @debug "This decomposition is running on Thread $(Threads.threadid())"
    global accelerators_vector
    set_csr_mat(sparse_mat)

    for decomp in system_matrix
        pop!(system_matrix) # clear system_matrix
        @debug "Cleared system_matrix"
    end

    decomps = Vector{AbstractLUdecomp}()
    # TODO: Measure overhead of setting accelerator device explicitly in every step
    # and try to remove if possible...
    Accelerators.set_acceleratordevice!(current_accelerator)

    if varDict["runtime_switch"]
        for accelerator in accelerators_vector
            # Only perform decomposition once, if multiple accelerators of the same type are present
            if any(x -> typeof(x) == get_ludecomp_type(accelerator), decomps)
                continue
            end
            lu_decomp = Accelerators.mna_decomp(sparse_mat, accelerator)
            push!(decomps, lu_decomp)
        end
    else                        # only calculate decomposition for the current accelerator

        lu_decomp = Accelerators.mna_decomp(sparse_mat, current_accelerator)
        push!(decomps, lu_decomp)
    end
    return decomps
end

function mna_solve(my_system_matrix, rhs)
    # Allow printing accelerator without debug statements
    (haskey(ENV, "JL_MNA_PRINT_ACCELERATOR") && ENV["JL_MNA_PRINT_ACCELERATOR"] == "true" ?
        println(typeof(current_accelerator))
        : nothing)
    
    Accelerators.set_acceleratordevice!(current_accelerator)        # sets the ACTUAL physical accelerator device
    idx = findfirst(x -> typeof(x) == get_ludecomp_type(current_accelerator), my_system_matrix) 

    
    if idx === nothing
        @debug "Decomposition for $(typeof(current_accelerator)) is not valid, recalculating..."
        global csr_mat
        lu_decomp = Accelerators.mna_decomp(csr_mat, current_accelerator) # Recalculate decomposition if not valid
        push!(my_system_matrix, lu_decomp)
        sys_mat = lu_decomp
    else
        sys_mat = my_system_matrix[idx]
    end


    @debug "Using system matrix of type $(typeof(sys_mat)) for solving."
    return Accelerators.mna_solve(sys_mat, rhs, current_accelerator)
end
mna_solve(system_matrix, rhs, accelerator::DummyAccelerator) = mna_solve(system_matrix, rhs, NoAccelerator())
