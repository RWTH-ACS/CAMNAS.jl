# Main module providing ccallable wrapper functions for the mna solver library
module CAMNAS

export get_mna_plugin, init, decomp, solve, cleanup, log
export dpsim_csr_matrix
export systemCheck

using SparseMatricesCSR

using CUDA

include("mna_solver.jl")


"""
    mutable struct dpsim_csr_matrix

Struct representation of the DPsim sparse matrix in CSR format. It is used to represent the system matrix for the mna solver library.

Fields:
- `values::Ptr{Cdouble}`    # size: nnz
- `rowIndex::Ptr{Cint}`     # size: row_number + 1
- `colIndex::Ptr{Cint}`     # size: nnz
- `row_number::Cint`        # number of rows of the matrix
- `nnz::Cint`               # number of non-zero elements in the matrix
"""
mutable struct dpsim_csr_matrix
    values::Ptr{Cdouble}    # size: nnz
    rowIndex::Ptr{Cint}     # size: row_number + 1
    colIndex::Ptr{Cint}     # size: nnz
    row_number::Cint        # number of rows of the matrix
    nnz::Cint               # number of non-zero elements in the matrix
end

function Base.size(mat::dpsim_csr_matrix)
    return (mat.row_number, mat.row_number)
end

"""
    @callable init(matrix_ptr::Ptr{dpsim_csr_matrix})::Cint

Initialize the MNA solver with the given system matrix. The matrix will be converted to a Julia SparseMatrix in CSR format and decomposed using LU factorization.

The LU factorization is internally stored as `system_matrix` and implicitly used in the solve function.
"""
function init end # Dummy function to allow documentation for ccallable function
Base.@ccallable function init(matrix_ptr::Ptr{dpsim_csr_matrix})::Cint
    sparse_mat = mat_ctojl(matrix_ptr)
    mna_init(sparse_mat)

    lu_mat = mna_decomp(sparse_mat)
    #@debug "Current LU factorization: $lu_mat"
    global system_matrix = lu_mat

    return 0
end

"""
    @callable decomp(matrix_ptr::Ptr{dpsim_csr_matrix})::Cint

Decompose the given system matrix using LU factorization.

The new LU factorization is internally stored as `system_matrix` and implicitly used in the solve function.
"""
function decomp end # Dummy function to allow documentation for ccallable function
Base.@ccallable function decomp(matrix_ptr::Ptr{dpsim_csr_matrix})::Cint
    sparse_mat = mat_ctojl(matrix_ptr)

    lu_mat = mna_decomp(sparse_mat)
    global system_matrix = lu_mat

    return 0
end

"""
    @callable solve(rhs_values_ptr::Ptr{Cdouble}, lhs_values_ptr::Ptr{Cdouble})::Cint

Solve the system `l = A \\ r` with the system matrix `A`, the given right-hand side vector `r` and store the result in the given left-hand side values pointer `l`.
"""
function solve end # Dummy function to allow documentation for ccallable function
Base.@ccallable function solve(rhs_values_ptr::Ptr{Cdouble}, lhs_values_ptr::Ptr{Cdouble})::Cint
    @debug @isdefined system_matrix
    (@isdefined system_matrix) || ( error("System matrix not initialized! Call init or decomp first!"); return -1 )
    
    # if typeof(system_matrix[1]) == DummyAccelerator_LUdecomp
    #     @debug typeof(system_matrix[2])
    #     @debug dump(system_matrix)
    #     lu_decomp = transfer_LU_CUDA2CPU(system_matrix[2])
    #     system_matrix[1] = lu_decomp
    #     @debug "This is cpu lu_decomp $(system_matrix[1])"
    # end

    dim = size(system_matrix[1].lu_decomp)[1] # Matrix is quadratic, so we can use m or n
    rhs = unsafe_wrap(Array, rhs_values_ptr, dim)

    @debug "rhs = $rhs"
    @debug "lhs = $(unsafe_wrap(Array, lhs_values_ptr, dim))"

    result = mna_solve(system_matrix, rhs)

    @debug "result = $result | $(typeof(result))"

    unsafe_copyto!(lhs_values_ptr, pointer(result), dim)

    return 0
end

"""
    @callable cleanup()::Cvoid

Cleanup the MNA solver.

__Currently not implemented...__
"""
function cleanup end # Dummy function to allow documentation for ccallable function
Base.@ccallable function cleanup()::Cvoid
    #mna_cleanup()
end

"""
    @callable log(log_string::Cstring)::Cvoid

Print the given log string to the console.
"""
function log end # Dummy function to allow documentation for ccallable function
Base.@ccallable function log(log_string::Cstring)::Cvoid
    println("[Log]: $(unsafe_string(log_string))")
end


"""
    mat_ctojl(matrix_ptr::Ptr{dpsim_csr_matrix})

    Convert a `dpsim_sparse_csr` pointer to a Julia `SparseMatrixCSR` with indexing starting at 0.
"""
function mat_ctojl(matrix_ptr::Ptr{dpsim_csr_matrix})
    mat_ptr = unsafe_load(matrix_ptr)
    @debug "mat_ptr = $mat_ptr"

    sparse_mat = SparseMatrixCSR{0}(
        mat_ptr.row_number, # Number of rows
        mat_ptr.row_number, # Matrices are square
        unsafe_wrap(Array, mat_ptr.rowIndex, mat_ptr.row_number+1), # Row pointers
        unsafe_wrap(Array, mat_ptr.colIndex, mat_ptr.nnz), # Column indices
        unsafe_wrap(Array, mat_ptr.values, mat_ptr.nnz) # Non-zero values
    )
    @debug "sparse_mat = $(dump(sparse_mat))"

    return sparse_mat
end
end # module
