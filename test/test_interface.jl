
begin # Initialization
    ##############################################################
    ## Use this varibale to define the size of the input files ##
    ##############################################################
    const inputSize = "small" # small, medium, big
    ##############################################################

    @assert inputSize in ["small", "medium", "big"]
    ENV["JULIA_DEBUG"] = "CAMNAS" # Enable debug output
    ENV["JL_MNA_RUNTIME_SWITCH"] = "true" # Enable runtime switch
    ENV["JL_MNA_PRINT_ACCELERATOR"] = "true" # Enable printing accelerator in each solve steps
    push!(LOAD_PATH, pwd())
    #push!(LOAD_PATH, "$(pwd())/accelerators")
    @info LOAD_PATH
    using Pkg
    Pkg.activate(LOAD_PATH[4])
    Pkg.status()

    using CAMNAS
    using Profile

    struct ArrayPath path::String end
    struct VectorPath path::String end

    function read_input(path::ArrayPath)
        # Read system matrix from file
        system_matrix_strings = readlines(path.path)

        # Sanize strings
        system_matrix_strings = replace.(system_matrix_strings, r"[\[\],]" => "")

        # Convert system to dpsim_csr_matrix
        values = parse.(Float64, split(system_matrix_strings[1]))
        rowIndex = parse.(Cint, split(system_matrix_strings[2]))
        colIndex = parse.(Cint, split(system_matrix_strings[3]))

        system_matrix = CAMNAS.dpsim_csr_matrix(
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

    GC.enable(false) # We cannot be sure that system_matrix is garbage collected before the pointer is passed...
    system_matrix = read_input(ArrayPath("$(@__DIR__)/system_matrix_$inputSize.txt"))
    system_matrix_ptr = pointer_from_objref(system_matrix)
    rhs_vector = read_input(VectorPath("$(@__DIR__)/rhs_$inputSize.txt"))
    lhs_vector = zeros(Float64, length(rhs_vector))
    rhs_reset = ones(Float64, length(rhs_vector))

    init(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
    GC.enable(true)
end # end Initialization

begin # Decomposition step
    GC.enable(false) # We cannot be sure that system_matrix is garbage collected before the pointer is passed...
    system_matrix = read_input(ArrayPath("$(@__DIR__)/system_matrix_$inputSize.txt"))
    system_matrix_ptr = pointer_from_objref(system_matrix)
    rhs_vector = read_input(VectorPath("$(@__DIR__)/rhs_$inputSize.txt"))
    lhs_vector = zeros(Float64, length(rhs_vector))
    rhs_reset = ones(Float64, length(rhs_vector))

    
    @time decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
    GC.enable(true)
end # end Decomposition

begin # Solving step 
    @time solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_reset), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
end # end Solving

begin # Cleanup step
    cleanup()
end # end Cleanup
