# build.jl


import PackageCompiler

using Pkg
using TOML
using Logging
using Test

const build_dir = @__DIR__
const target_dir = ARGS[1]

project = Base.active_project()
project_dir = dirname(project)

function require_without_errors(package_name)
    logger = TestLogger(min_level=Logging.Error)

    with_logger(logger) do
        Base.require(Main, Symbol(package_name))
    end

    if !isempty(logger.logs)
        error("Loading $package_name produced errors")
    end
end

# Note: Adding packages individually leads to incompatible versions. Thats they are tested individially and then addes as a list
function probe_weakdeps()
    toml = TOML.parsefile(project)
    weakdeps = get(toml, "weakdeps", Dict())

    addable = String[]
    original_project = Base.active_project()

    for (package_name, _) in weakdeps
        @info("Testing host for weak dependency $package_name support...")

        try
            # Note: We use a temp env because after adding the weakdep the extension will be loaded.
            # If the weakdep then gets removed it triggers a load error
            Pkg.activate(; temp = true)

            Pkg.add(package_name)

            # Check for error messages
            require_without_errors(package_name)

            # TODO: More checks like Cuda functional

            push!(addable, package_name)
        catch err
            try
                Pkg.rm(package_name)
            catch err
            end
            @warn "Could not add weak dependency $package_name; reason: $err" exception=(err, catch_backtrace())
        end
    end

    Pkg.activate(original_project)

    return addable
end

# Add packages for extension activation
# addable = probe_weakdeps()
#@info "Adding the following packages for extensions: $addable"
# Pkg.add(addable)
Pkg.add(["CUDA", "CUDSS"])

@info "Creating CAMNAS solver library in $target_dir"
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