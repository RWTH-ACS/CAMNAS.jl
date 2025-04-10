using SparseArrays
using CUDA.CUSPARSE
using CUSOLVERRF
using LinearAlgebra
using FileWatching

include("config.jl")

# Hardwareawareness
abstract type AbstractAccelerator end
struct NoAccelerator <: AbstractAccelerator end
struct CUDAccelerator <: AbstractAccelerator end
struct DummyAccelerator <: AbstractAccelerator end

system_environment = Channel(1)
accelerator = NoAccelerator()
system_matrix = nothing


function find_accelerator()
    if varDict["allow_gpu"] && has_cuda()
        @debug "CUDA available! Try using CUDA accelerator..."
        try
            CuArray(ones(1))
            accelerator = CUDAccelerator()
            @info "[CAMNAS] CUDA driver available and CuArrays package loaded. Using CUDA accelerator..."
        catch e
            @warn "CUDA driver available but could not load CuArrays package."
        end
    elseif !@isdefined accelerator
        @info "[CAMNAS] No accelerator found."
        accelerator = NoAccelerator()
    end
    return accelerator
end

function systemcheck()
    if varDict["hwAwarenessDisabled"]
        @info "[CAMNAS] Hardware awareness disabled... Using Fallback implementation"
        return NoAccelerator()
    else
        return find_accelerator()
    end
end


function file_watcher()
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
    global accelerator
    while true
        val = take!(system_environment)
        @debug "Received new system environment!: $val"

        for line in split(val, '\n')[2:end]
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
        val === nothing ? break : nothing

        # Currently, we implement a GPU-favoring approach
        if varDict["allow_gpu"] && has_cuda()
            typeof(accelerator) == CUDAccelerator || set_accelerator!(CUDAccelerator())
        elseif varDict["allow_cpu"]
            typeof(accelerator) == DummyAccelerator || set_accelerator!(DummyAccelerator())
        else
            typeof(accelerator) == NoAccelerator || set_accelerator!(NoAccelerator())
        end

        @info "[CAMNAS] Currently used accelerator: $(typeof(accelerator))"
    end
    @debug "Accelerator determination stopped!"
end

function set_accelerator!(acc)
    @debug "Setting accelerator to: $(typeof(acc))"
    global accelerator = acc
end

# Housekeeping
function mna_init(sparse_mat)
    global varDict = parse_env_vars()
    create_env_file()

    global accelerator = systemcheck()
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

function mna_decomp(sparse_mat, accelerator::AbstractAccelerator)
    lu_decomp = SparseArrays.lu(sparse_mat)
    @debug "CPU $lu_decomp"
    return lu_decomp
end

function mna_decomp(sparse_mat, accelerator::DummyAccelerator)
    lu_decomp = SparseArrays.lu(sparse_mat)
    @debug "Dummy"
    return lu_decomp
end

function mna_decomp(sparse_mat, accelerator::CUDAccelerator)
    matrix = CuSparseMatrixCSR(CuArray(sparse_mat)) # Sparse GPU implementation
    lu_decomp = CUSOLVERRF.RFLU(matrix; symbolic=:RF)
    return lu_decomp
end

function mna_decomp(sparse_mat)
    set_csr_mat(sparse_mat)
    if varDict["fast_switch"]
        return [mna_decomp(sparse_mat, NoAccelerator()), mna_decomp(sparse_mat, CUDAccelerator())]
    end
    return [mna_decomp(sparse_mat, accelerator), nothing]
end

function mna_solve(system_matrix, rhs, accelerator::AbstractAccelerator)
    return system_matrix \ rhs
end

function mna_solve(system_matrix, rhs, accelerator::CUDAccelerator)
    rhs_d = CuVector(rhs)
    ldiv!(system_matrix, rhs_d)
    return Array(rhs_d)
end

function mna_solve(my_system_matrix, rhs)

    # Allow printing accelerator without debug statements
    (haskey(ENV, "PRINT_ACCELERATOR") && ENV["PRINT_ACCELERATOR"] == "true" ?
        println(typeof(accelerator))
        : nothing)
    (typeof(accelerator) == CUDAccelerator && varDict["fast_switch"]) ? sys_mat = my_system_matrix[2] : sys_mat = my_system_matrix[1]

    return mna_solve(sys_mat, rhs, accelerator)
end
mna_solve(system_matrix, rhs, accelerator::DummyAccelerator) = mna_solve(system_matrix, rhs, NoAccelerator())