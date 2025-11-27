module Utils

using CAMNAS
using SparseMatricesCSR

struct ArrayPath
    path::String
end

struct VectorPath
    path::String
end

function to_zerobased_csr(matrix)
    csr = SparseMatrixCSR(matrix)
    csr.colval .-= 1  # Convert column indices to 0-based
    csr.rowptr .-= 1  # Convert row pointers to 0-based
    return csr
end

function csr_to_dpsim(csr::SparseMatrixCSR)
    matrix = dpsim_csr_matrix(
        Base.unsafe_convert(Ptr{Cdouble}, csr.nzval),
        Base.unsafe_convert(Ptr{Cint}, convert(Array{Int32}, csr.rowptr)), #! Cint expects 32 bit value
        Base.unsafe_convert(Ptr{Cint}, convert(Array{Int32}, csr.colval)),
        Int32(csr.m),
        Int32(length(csr.nzval))
    )
end

function read_input(path::ArrayPath)
    # Read system matrix from file
    system_matrix_strings = readlines(path.path)

    # Sanize strings
    system_matrix_strings = replace.(system_matrix_strings, r"[\[\],]" => "")

    # Convert system to dpsim_csr_matrix
    values = parse.(Float64, split(system_matrix_strings[1]))
    rowIndex = parse.(Cint, split(system_matrix_strings[2]))
    colIndex = parse.(Cint, split(system_matrix_strings[3]))

    system_matrix = dpsim_csr_matrix(
        Base.unsafe_convert(Ptr{Cdouble}, values),
        Base.unsafe_convert(Ptr{Cint}, rowIndex),
        Base.unsafe_convert(Ptr{Cint}, colIndex),
        parse(Int32, system_matrix_strings[4]),
        parse(Int32, system_matrix_strings[5])
    )

    return system_matrix
end

function read_input(path::VectorPath)
    # Reard right hand side vector from file
    rhs_vector_strings = readlines(path.path)

    # Sanitize rhs strings and parse into Float64 vector
    rhs_vector_strings = replace.(rhs_vector_strings, r"[\[\],]" => "")
    rhs_vector = parse.(Float64, split(rhs_vector_strings[1]))
end

end