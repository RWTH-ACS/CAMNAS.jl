# build.jl


import PackageCompiler

const build_dir = @__DIR__
const target_dir = ARGS[1]

# FIXME: Remove once the issue with PackageCompiler and Julia v1.11. is resolved.
# See: https://github.com/JuliaLang/PackageCompiler.jl/issues/990
delete!(ENV, "JULIA_NUM_THREADS")

# Add packages for extension activation
# See https://github.com/RWTH-ACS/CAMNAS.jl/issues/53 Fix#53
# For example uncomment:
# Pkg.add(["CUDA", "CUDSS"])

println("Creating CAMNAS solver library in $target_dir")
PackageCompiler.create_library("$(build_dir)/..", target_dir;
                                lib_name="camnasjl",
                                precompile_execution_file="$(@__DIR__)/precompile_statements.jl",
                                incremental=false,
                                filter_stdlibs=false,
                                include_lazy_artifacts=true,
                                header_files = ["$(@__DIR__)/camnasjl.h"],
                                force=true,
                                cpu_target="native"
                            )