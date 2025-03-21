using CUDA
try
    has_cuda() ? CuArray(ones(1)) : nothing
catch e
    @warn "CUDA not available. Resume precompilation..."
end