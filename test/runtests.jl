using CAMNAS, Test
using CUDA

@testset "CAMNAS" begin

    # Read and convert test inputs
    sys_mat_str = readlines("system_matrix_small.txt")
    sys_mat_str = replace.(sys_mat_str, r"[\[\],]" => "")

    values = parse.(Float64, split(sys_mat_str[1]))
    rowIndex = parse.(Cint, split(sys_mat_str[2]))
    colIndex = parse.(Cint, split(sys_mat_str[3]))

    sys_mat = dpsim_csr_matrix(
        Base.unsafe_convert(Ptr{Cdouble}, values),
        Base.unsafe_convert(Ptr{Cint}, rowIndex),
        Base.unsafe_convert(Ptr{Cint}, colIndex),
        parse(Int32, sys_mat_str[4]),
        parse(Int32, sys_mat_str[5])
    )
    sys_mat_ptr = pointer_from_objref(sys_mat)

    rhs_vec_strings = readlines("rhs_small.txt")
    rhs_vec_strings = replace.(rhs_vec_strings, r"[\[\],]" => "")
    rhs_vec = parse.(Float64, split(rhs_vec_strings[1]))

    lhs_vec = zeros(Float64, length(rhs_vec))

    @testset "Initialization" begin
        @test init(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, sys_mat_ptr)) == 0
    end 

    @testset "Solving" begin
        @test solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vec), Base.unsafe_convert(Ptr{Cdouble}, lhs_vec)) == 0
        @test lhs_vec == [1.0, 1/2, 1/3]
    end

    @testset "Decomposition" begin
        # Change system matrix values
        sys_mat_bak = sys_mat.values
        sys_mat.values = Base.unsafe_convert(Ptr{Cdouble}, [4.0, 5.0, 6.0])

        # Decompose new system matrix
        @test decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, sys_mat_ptr)) == 0

        # Solve system with new matrix
        @test solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vec), Base.unsafe_convert(Ptr{Cdouble}, lhs_vec)) == 0
        @test lhs_vec == [1/4, 1/5, 1/6]

        # Restore system matrix values
        sys_mat.values = sys_mat_bak
    end        

    @testset "Generator" begin
        pre_seed = 1337

        include("Generator.jl")
        include("MatrixValidator.jl")

        settings = Generator.Settings(
            dimension=300,
            density=0.1,
            magnitude_off=0.05,
            delta=0.5,
            seed=pre_seed
        )

        matrix = Generator.generate_matrix(settings)

        # Dimensions
        @test MatrixValidator.is_quadratic(matrix)
        @test MatrixValidator.m(matrix) == settings.dimension
        @test MatrixValidator.n(matrix) == settings.dimension

        # Density
        density = MatrixValidator.density(matrix)
        density_tolerance = 0.01
        @test isapprox(density, settings.density; atol=density_tolerance)

        # Condition
        condition_tresh = 2
        @test MatrixValidator.condition(matrix) < condition_tresh

        # LU-decomposability
        @test MatrixValidator.is_lu_decomposable(matrix)

        # Solving and rhs generation
        rhs = Generator.generate_rhs_vector(matrix; prefered_solution=ones(Float64, size(matrix, 1)))
        x = matrix \ rhs
        solving_tolerance = 1e-8
        @test all(value -> isapprox(value, 1.0; atol=solving_tolerance), x)

        # Random seed
        @test matrix == Generator.generate_matrix(settings) # Reproducability

        new_settings = Generator.Settings(
            dimension=settings.dimension,
            density=settings.density,
            magnitude_off=settings.magnitude_off,
            delta=settings.delta,
            seed=settings.seed + 1
        )

        @test matrix != Generator.generate_matrix(new_settings) # Randomness

    end

    @testset "Benchmark" begin
        pre_seed = 1337
        include("Benchmark.jl")
        include("Generator.jl")

        # from files
        @test Benchmark.benchmark("system_matrix_small.txt", "rhs_small.txt") isa Benchmark.BenchmarkResult

        # generated matrix
        settings = Generator.Settings(
            dimension=300,
            density=0.01,
            seed=pre_seed
        )
        matrix = Generator.generate_matrix(settings)
        rhs_vector = Generator.generate_rhs_vector(matrix)
        @test Benchmark.benchmark(matrix, rhs_vector) isa Benchmark.BenchmarkResult

        # CUDA accelerator
        ENV["JL_MNA_RUNTIME_SWITCH"] = "true"
        ENV["JL_MNA_FORCE_GPU"] = "true"
        @test Benchmark.benchmark(matrix, rhs_vector) isa Benchmark.BenchmarkResult skip=!CUDA.has_cuda_gpu()

    end
end # testset "CAMNAS"

