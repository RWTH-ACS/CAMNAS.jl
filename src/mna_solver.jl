using SparseArrays
using CUDA.CUSPARSE
using CUSOLVERRF
using LinearAlgebra
using FileWatching
using TOML

include("accelerators/Accelerators.jl")
include("config.jl")


using .Accelerators






# Accelerator Selection
abstract type AbstractSelectionStrategy end

struct DefaultStrategy <: AbstractSelectionStrategy end #choose first accelerator from a defined order
struct LowestPowerStrategy <: AbstractSelectionStrategy end
struct HighestPerfStrategy <: AbstractSelectionStrategy end
struct NoStrategy <: AbstractSelectionStrategy end


# Vector of available accelerators
global accelerators_vector = Vector{AbstractAccelerator}()
system_environment = Channel(1)
accelerator = NoAccelerator()
current_strategy = NoStrategy()
system_matrix = Vector{AbstractLUdecomp}()   



function select_strategy(strategy::AbstractSelectionStrategy, accelerators_vector::Vector{AbstractAccelerator})
    global accelerators_vector
    @debug "Strategy not implemented, falling back to DefaultStrategy"
    select_strategy(DefaultStrategy(), accelerators_vector)
end

function select_strategy(strategy::DefaultStrategy, accelerators_vector::Vector{AbstractAccelerator})
    # sort vector of accelerators to a specific order and then choose the first available
    global current_strategy = strategy
    allowed = Vector{AbstractAccelerator}()

    # This really doesnt seem nice
    if varDict["allow_gpu"]
        allowed = filter(x -> typeof(x) != NoAccelerator, accelerators_vector)
    end
    if varDict["allow_cpu"]
        push!(allowed, accelerators_vector[findfirst(x -> typeof(x) == NoAccelerator, accelerators_vector)])
    end
    if isempty(allowed)
        @error "No accelerators available for selection."
        return nothing
    end

    idx = 1         # choose first accelerator from the available accelerators_vector
    set_accelerator!(allowed[idx])
    @debug "DefaultStrategy selected, using $(accelerators_vector[idx])"
end

function select_strategy(strategy::NoStrategy, accelerators_vector::Vector{AbstractAccelerator})
    # sort vector of accelerators to a specific order and then choose the first available
    global current_strategy = strategy
    @debug "NoStrategy selected"
end

function select_strategy(strategy::LowestPowerStrategy, accelerators_vector::Vector{AbstractAccelerator})
    global accelerators_vector
    global current_strategy = strategy
    # available = filter(x -> x.properties.availability, accelerators_vector)
    # value, index = findmin(x -> x.properties.power_watts, available)
    allowed = Vector{AbstractAccelerator}()

    if !varDict["allow_gpu"] && !varDict["allow_cpu"]
        @error "Nothing allowed"
        return nothing
    end 

    # This really doesnt seem nice
    if varDict["allow_gpu"]
        allowed = filter(x -> typeof(x) != NoAccelerator, accelerators_vector)
    end
    if varDict["allow_cpu"]
        push!(allowed, accelerators_vector[findfirst(x -> typeof(x) == NoAccelerator, accelerators_vector)])
    end
    if isempty(allowed)
        @error "No accelerators available for selection."
        return nothing
    end

    value, index = findmin(x -> x.properties.power_watts, allowed)
    set_accelerator!(allowed[index])
    


end

function select_strategy(strategy::HighestPerfStrategy, accelerators_vector::Vector{AbstractAccelerator})
    global accelerators_vector
    global current_strategy = strategy
    # available = filter(x -> x.properties.availability, accelerators_vector)
    # value, index = findmax(x -> x.properties.perf, available)
    allowed = Vector{AbstractAccelerator}()

    if !varDict["allow_gpu"] && !varDict["allow_cpu"]
        @error "Nothing allowed"
        return nothing
    end 

    # This really doesnt seem nice
    if varDict["allow_gpu"]
        allowed = filter(x -> typeof(x) != NoAccelerator, accelerators_vector)
    end
    if varDict["allow_cpu"]
        push!(allowed, accelerators_vector[findfirst(x -> typeof(x) == NoAccelerator, accelerators_vector)])
    end
    if isempty(allowed)
        @error "No accelerators available for selection."
        return nothing
    end

    value, index = findmax(x -> x.properties.performanceIndicator, allowed)
    set_accelerator!(allowed[index])
    

end



function find_accelerator()
    global accelerators_vector
    try
        @debug "Trying to load accelerators..."
        Accelerators.load_all_accelerators(accelerators_vector)
    catch e 
        @error "Failed to load accelerators: $e"
        set_accelerator!(NoAccelerator()) 
    end
    
    evaluateSystemEnv(nothing)

    # if !isempty(accelerators_vector) && varDict["allow_gpu"]
    #     idx = findfirst(x -> typeof(x) != NoAccelerator, accelerators_vector)
    #     set_accelerator!(accelerators_vector[idx])
    # elseif !@isdefined accelerator
    #     @info "[CAMNAS] No accelerator found."
    #     set_accelerator!(NoAccelerator())
    # end

    @debug "Present accelerators: $([a.name for a in accelerators_vector])"

    
end


function systemcheck()
    if varDict["hwAwarenessDisabled"]
        @info "[CAMNAS] Hardware awareness disabled... Using Fallback implementation"
        set_accelerator!(NoAccelerator())
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
        
        evaluateSystemEnv(val)
        # for line in split(val, '\n')[2:end]
        #     if length(line) == 0
        #         continue
        #     end
        #     key, value = split(line)
        #     if key == "allow_cpu"
        #         allow_cpu = parse(Bool, value)
        #         varDict["allow_cpu"] = allow_cpu
        #     elseif key == "allow_gpu"
        #         allow_gpu = parse(Bool, value)
        #         varDict["allow_gpu"] = allow_gpu
        #     else
        #         varDict[key] = parse(Bool, value)
        #     end
        # end

        # @debug "Allow CPU is: $(varDict["allow_cpu"])"
        # @debug "Allow GPU is: $(varDict["allow_gpu"])"
        # @debug "$varDict"

        # # Stop accelerator determination if nothing-value is received
        # val === nothing ? break : nothing

        # # Currently, force statments are the strongest, then consider strategies
        # if varDict["runtime_switch"]
        #     if varDict["force_cpu"] || varDict["force_gpu"]

        #         # FORCING
        #         if varDict["force_cpu"] && varDict["force_gpu"]
                
        #             @debug "Conflict: Both 'force_cpu' and 'force_gpu' are set. Only one can be forced."
        #             idx = findfirst(x -> x.name == "cpu", accelerators_vector)
        #             typeof(accelerator) == NoAccelerator || set_accelerator!(accelerators_vector[idx])
                
        #         elseif varDict["allow_gpu"] && varDict["force_gpu"] # anything but cpu is considered gpu
                
        #             idx = findfirst(x -> typeof(x) != NoAccelerator, accelerators_vector)   
        #             set_accelerator!(accelerators_vector[idx])
                
        #         elseif varDict["allow_cpu"] && varDict["force_cpu"]
        #             idx = findfirst(x -> x.name == "cpu", accelerators_vector)
        #             typeof(accelerator) == NoAccelerator || set_accelerator!(accelerators_vector[idx])
        #         end
        #         @debug "Forcing prioritized, using NoStrategy"
        #         select_strategy(NoStrategy(), accelerators_vector)

        #     elseif varDict["allow_strategies"]
        #         # STRATEGIES
        #         if varDict["highest_flop_strategy"] && varDict["lowest_power_strategy"]
        #             @debug "Too many Stragegies set! Only one can be used at a time."
        #             select_strategy(DefaultStrategy(), accelerators_vector)    

        #         elseif varDict["highest_flop_strategy"]
        #             @debug "Selected HighestPerfStrategy"
        #             select_strategy(HighestPerfStrategy(), accelerators_vector)
                
        #         elseif varDict["lowest_power_strategy"] 
        #             @debug "Selected LowestPowerStrategy"
        #             select_strategy(LowestPowerStrategy(), accelerators_vector)
        #         else
        #             @debug "Selected DefaultStrategy"
        #             select_strategy(DefaultStrategy(), accelerators_vector)
        #         end
        #     elseif varDict["allow_gpu"] 
        #         idx = findfirst(x -> typeof(x) != NoAccelerator, accelerators_vector)   
        #         set_accelerator!(accelerators_vector[idx])
        #         @debug "No strategy selected, using NoStrategy"
        #         select_strategy(NoStrategy(), accelerators_vector)
            
        #     elseif varDict["allow_cpu"]
        #         idx = findfirst(x -> x.name == "cpu", accelerators_vector)
        #         typeof(accelerator) == NoAccelerator || set_accelerator!(accelerators_vector[idx])
        #         @debug "No strategy selected, using NoStrategy"
        #         select_strategy(NoStrategy(), accelerators_vector)
            
        #     else
        #         @debug "Conflict: Nothing is allowed. THIS DOESNT MAKE SENSE!"
        #     end

        #     @info "[CAMNAS] Currently used accelerator: $accelerator" 
        #     if varDict["allow_strategies"]
        #         @info "[CAMNAS] Currently used strategy: $(typeof(current_strategy))"
        #     end
        # else
        #     @debug "Runtime switch is disabled, Accelerator will not be changed."
        # end
    end
    @debug "Accelerator determination stopped!"
end

function evaluateSystemEnv(content)
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
        key, value = split(line)
        if key == "allow_cpu"
            allow_cpu = parse(Bool, value)
            varDict["allow_cpu"] = allow_cpu
        elseif key == "allow_gpu"
            allow_gpu = parse(Bool, value)
            varDict["allow_gpu"] = allow_gpu
        else
            varDict[key] = parse(Bool, value)
        end
    end

    @debug "Allow CPU is: $(varDict["allow_cpu"])"
    @debug "Allow GPU is: $(varDict["allow_gpu"])"
    @debug "$varDict"

    # Stop accelerator determination if nothing-value is received
    #content === nothing ? (return) : nothing

    # Currently, force statments are the strongest, then consider strategies
    if varDict["runtime_switch"] || first_run
        if varDict["force_cpu"] || varDict["force_gpu"]

            # FORCING
            if varDict["force_cpu"] && varDict["force_gpu"]
            
                @debug "Conflict: Both 'force_cpu' and 'force_gpu' are set. Only one can be forced."
                idx = findfirst(x -> x.name == "cpu", accelerators_vector)
                typeof(accelerator) == NoAccelerator || set_accelerator!(accelerators_vector[idx])
            
            elseif varDict["allow_gpu"] && varDict["force_gpu"] # anything but cpu is considered gpu
            
                idx = findfirst(x -> typeof(x) != NoAccelerator, accelerators_vector)   
                set_accelerator!(accelerators_vector[idx])
            
            elseif varDict["allow_cpu"] && varDict["force_cpu"]
                idx = findfirst(x -> x.name == "cpu", accelerators_vector)
                typeof(accelerator) == NoAccelerator || set_accelerator!(accelerators_vector[idx])
            end
            @debug "Forcing prioritized, using NoStrategy"
            select_strategy(NoStrategy(), accelerators_vector)

        elseif varDict["allow_strategies"]
            # STRATEGIES
            if varDict["highest_flop_strategy"] && varDict["lowest_power_strategy"]
                @debug "Too many Stragegies set! Only one can be used at a time."
                select_strategy(DefaultStrategy(), accelerators_vector)    

            elseif varDict["highest_flop_strategy"]
                @debug "Selected HighestPerfStrategy"
                select_strategy(HighestPerfStrategy(), accelerators_vector)
            
            elseif varDict["lowest_power_strategy"] 
                @debug "Selected LowestPowerStrategy"
                select_strategy(LowestPowerStrategy(), accelerators_vector)
            else
                @debug "Selected DefaultStrategy"
                select_strategy(DefaultStrategy(), accelerators_vector)
            end
        elseif varDict["allow_gpu"] 
            idx = findfirst(x -> typeof(x) != NoAccelerator, accelerators_vector)   
            set_accelerator!(accelerators_vector[idx])
            @debug "No strategy selected, using NoStrategy"
            select_strategy(NoStrategy(), accelerators_vector)
        
        elseif varDict["allow_cpu"]
            idx = findfirst(x -> x.name == "cpu", accelerators_vector)
            typeof(accelerator) == NoAccelerator || set_accelerator!(accelerators_vector[idx])
            @debug "No strategy selected, using NoStrategy"
            select_strategy(NoStrategy(), accelerators_vector)
        
        else
            @debug "Conflict: Nothing is allowed. THIS DOESNT MAKE SENSE!"
        end

        @info "[CAMNAS] Currently used accelerator: $accelerator" 
        if varDict["allow_strategies"]
            @info "[CAMNAS] Currently used strategy: $(typeof(current_strategy))"
        end
    else
        @debug "Runtime switch is disabled, Accelerator will not be changed."
    end
end    

# FIXME: this seems weird not to be in Accelerators.jl
function set_accelerator!(acc::AbstractAccelerator) 
    @debug "Setting accelerator to: $(typeof(acc))"
    global accelerator = acc
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
    @debug "This decomposition is running on $(Threads.threadid())"
    global accelerators_vector
    set_csr_mat(sparse_mat)

    for decomp in system_matrix
        pop!(system_matrix) # clear system_matrix
        @debug "Cleared system_matrix"
    end

    decomps = Vector{AbstractLUdecomp}()

    Accelerators.set_acceleratordevice!(accelerator)

    if varDict["runtime_switch"]
        for accelerator in accelerators_vector
            if any(x -> typeof(x) == get_ludecomp_type(accelerator), decomps) # check if accelerator is already in decomps
                continue
            end
            lu_decomp = Accelerators.mna_decomp(sparse_mat, accelerator)
            push!(decomps, lu_decomp)
        end
    else                        # only calculate decomposition for the current accelerator

        lu_decomp = Accelerators.mna_decomp(sparse_mat, accelerator)
        push!(decomps, lu_decomp)
    end
    return decomps
end





function mna_solve(my_system_matrix, rhs)

    # Allow printing accelerator without debug statements
    (haskey(ENV, "JL_MNA_PRINT_ACCELERATOR") && ENV["JL_MNA_PRINT_ACCELERATOR"] == "true" ?
        println(typeof(accelerator))
        : nothing)
    
    idx = findfirst(x -> typeof(x) == get_ludecomp_type(accelerator), my_system_matrix) 

    
    if idx === nothing
        @debug "Decomposition for $(typeof(accelerator)) is not valid, recalculating..."
        global csr_mat
        lu_decomp = Accelerators.mna_decomp(csr_mat, accelerator) # Recalculate decomposition if not valid
        push!(my_system_matrix, lu_decomp)
        sys_mat = lu_decomp
    else
        sys_mat = my_system_matrix[idx]
    end

    Accelerators.set_acceleratordevice!(accelerator)        # sets the ACTUAL physical accelerator device

    @debug "Using system matrix of type $(typeof(sys_mat)) for solving."
    return Accelerators.mna_solve(sys_mat, rhs, accelerator)
end
mna_solve(system_matrix, rhs, accelerator::DummyAccelerator) = mna_solve(system_matrix, rhs, NoAccelerator())

function transfer_LU_CUDA2CPU(cuda_lu::CUDAccelerator_LUdecomp) #transfer LU factorization from CUSOLVERRF.RFLU to SparseArrays.UMFPACK.UMFPACKLU type
    # Access combined LU matrix (GPU, CSR format)
    M_gpu = cuda_lu.lu_decomp.M    # M = L + U
    
    rowPtr = collect(M_gpu.rowPtr)
    colVal = collect(M_gpu.colVal)
    nzVal = collect(M_gpu.nzVal)

    nrow = size(M_gpu, 1)
    ncol = size(M_gpu, 2)

    # Construct CPU-side sparse matrix in CSR format
    M_cpu = SparseMatrixCSR{1}(nrow, ncol, rowPtr, colVal, nzVal)   # 1 indicates index base
    cpu_lu_decomp = SparseArrays.lu(M_cpu) |> NoAccelerator_LUdecomp
    
    @debug "Type of lu_decomp is $(typeof(lu_decomp))"
    return cpu_lu_decomp
end
