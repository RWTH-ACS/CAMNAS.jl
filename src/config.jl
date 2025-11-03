export create_env_file
export parse_env_vars

function env(env_var, default)
    if haskey(ENV, env_var)
        @info env_var * " = " * ENV[env_var]
        parse(typeof(default), ENV[env_var])
    else
        @info env_var * " not set. Using default value: " * string(default)
        default
    end
end

function create_env_file()
    filepath = "$(@__DIR__)/system.env"
    @info filepath
    println("Creating system environment file at $filepath")
    open(filepath, "w") do f
        println(f, "# This file is machine-generated - editing it directly is not advised\n")
        for (k, v) in varDict
            println(f, "$k $v")
        end
    end
end

function parse_env_vars()
    varDict = Dict()

    varDict["hwAwarenessDisabled"] = env("JL_MNA_DISABLE_AWARENESS", false)

    varDict["force_gpu"] = env("JL_MNA_FORCE_GPU", false)
    varDict["force_cpu"] = env("JL_MNA_FORCE_CPU", false)
    (varDict["force_gpu"] && varDict["force_cpu"]) && error("Cannot force CPU and GPU at the same time.")

    varDict["allow_gpu"] = env("JL_MNA_ALLOW_GPU", true)
    varDict["allow_cpu"] = env("JL_MNA_ALLOW_CPU", true)
    (!varDict["allow_gpu"] && !varDict["allow_cpu"]) && error("Cannot forbid CPU and GPU at the same time.")

    varDict["runtime_switch"] = env("JL_MNA_RUNTIME_SWITCH", false)

    varDict["lowest_power_strategy"] = env("JL_MNA_LOWEST_POWER_STRATEGY", false)
    varDict["highest_flop_strategy"] = env("JL_MNA_HIGHEST_FLOP_STRATEGY", false)
    varDict["allow_strategies"] = env("JL_MNA_ALLOW_STRATEGIES", true)

    return varDict
end