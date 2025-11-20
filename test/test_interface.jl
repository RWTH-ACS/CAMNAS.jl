
begin # Initialization
    ##############################################################
    ## Use this varibale to define the size of the input files ##
    ##############################################################
    const inputType = "generated" # small, medium, big, generated
    ##############################################################

    @assert inputType in ["small", "medium", "big", "generated"]
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

    include("Utils.jl")

    if inputType == "generated"
        include("Generator.jl")

        # Generate test matrix
        generator_settings = Generator.Settings(dimension=3, density=0.01)
        matrix = Generator.generate_matrix(generator_settings)

        # matrix to file
        csr_matrix = Utils.to_zerobased_csr(matrix)
        Generator.matrix_to_file(csr_matrix)

        # rhs to file
        rhs_vector = Generator.generate_rhs_vector(matrix) # assign directly
        Generator.rhs_to_file(rhs_vector)
    end

    GC.enable(false) # We cannot be sure that system_matrix is garbage collected before the pointer is passed...
    system_matrix = Utils.read_input(Utils.ArrayPath("$(@__DIR__)/system_matrix_$inputType.txt"))
    system_matrix_ptr = pointer_from_objref(system_matrix)
    rhs_vector = Utils.read_input(Utils.VectorPath("$(@__DIR__)/rhs_$inputType.txt"))
    lhs_vector = zeros(Float64, length(rhs_vector))
    rhs_reset = ones(Float64, length(rhs_vector))

    init(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
    GC.enable(true)
end # end Initialization

begin # Decomposition step
    GC.enable(false) # We cannot be sure that system_matrix is garbage collected before the pointer is passed...
    system_matrix = Utils.read_input(Utils.ArrayPath("$(@__DIR__)/system_matrix_$inputType.txt"))
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
