using Random, Distributions
using LinearAlgebra
using Statistics, StatsBase
using HDF5
using Plots, Plots.PlotMeasures

include("functions_layer_1_nonparametrics_loads_data.jl")
include("functions_layer_2_nonparametrics_loads_data.jl")
include("input_parameters_loads_data.jl")
#include("plot_trajectories.jl")
gt_trajectory = get_gt_trajectory()

function sampler_HMM(draws)
    n_rates::Int64 = max_n_system_states^2
    mcmc_save_loads = zeros(Int64, draws, max_n_system_states)
    mcmc_save_rates::Matrix{Float64} = zeros(draws, n_rates)
    mcmc_save_mean_donor::Matrix{Float64} = zeros(draws, max_n_system_states)
    mcmc_save_mean_acceptor::Matrix{Float64} = zeros(draws, max_n_system_states)
    mcmc_log_posterior::Vector{Float64} = zeros(draws)
    mcmc_save_trajectory = zeros(Int64, draws, nbins)

    println(" Check for existing MCMC data...")
    flush(stdout)

    last_draw::Int64,
    mcmc_save_MAP_trajectory::Vector{Int64} =
        check_existing_mcmc_data(
            mcmc_save_loads,
            mcmc_save_rates,
            mcmc_save_mean_donor,
            mcmc_save_mean_acceptor,
            mcmc_log_posterior,
            mcmc_save_trajectory)

    println(" Done.")
    flush(stdout)

    local loads::Vector{Int64}
    local rates::Vector{Float64}
    local state_trajectory::Vector{Int64}

    if last_draw == 1
        println(" No existing MCMC data found.")
        flush(stdout)

        println(" Initializing all variables...")
        flush(stdout)

        loads, rates, state_trajectory,
        mean_donor,
        mean_acceptor = initialize_variables()

        println(" Done.")
        flush(stdout)

        mcmc_save_loads[last_draw, 1:max_n_system_states] = loads[:]
        mcmc_save_rates[last_draw, 1:n_rates] = rates[:]
        mcmc_log_posterior[last_draw] =
            get_log_posterior(loads, rates, mean_donor,
                mean_acceptor)

        mcmc_save_trajectory[1, :] = state_trajectory[:]
        mcmc_save_MAP_trajectory = copy(state_trajectory)
        mcmc_save_mean_donor[last_draw, :] = mean_donor
        mcmc_save_mean_acceptor[last_draw, :] = mean_acceptor
    else
        println(" Existing MCMC data found.")
        flush(stdout)

        loads = mcmc_save_loads[last_draw, :]
        rates = mcmc_save_rates[last_draw, :]
        mean_donor = mcmc_save_mean_donor[last_draw, :]
        mean_acceptor = mcmc_save_mean_acceptor[last_draw, :]

        state_trajectory = copy(mcmc_save_trajectory[last_draw, :])
    end

    println(" Get generator matrix, et cetera...")
    flush(stdout)

    generator, rho = get_generator(loads, rates)
    FRET_efficiencies = get_FRET_efficiencies(loads, rates)
    loads_active = filter(x -> x != 0, loads)
    n_system_states = size(loads_active)[1]

    println(" Done.")
    flush(stdout)

    temperature = 1.0 + starting_temperature * exp(-last_draw / 480)

    println("****************************************************************")
    @show last_draw
    @show loads, n_system_states
    @show temperature
    @show rates[1]
    @show rates[2]
    @show generator
    @show FRET_efficiencies
    @show mcmc_log_posterior[last_draw]

    println(" Starting Sampler...")
    flush(stdout)

    plot_everything(last_draw, state_trajectory, max_n_system_states,
        mcmc_log_posterior)

    for draw in last_draw+1:draws
        temperature = 1.0 + starting_temperature * exp(-draw / 300)

        # Get the new rates for conformation dynamics
        rates = sample_transition_rates(draw, loads, rates,
            state_trajectory, mean_donor,
            mean_acceptor,
            temperature)

        # Sample Parameters Governing Emissions
        rates, mean_donor,
        mean_acceptor = sample_emission_parameters(
            draw, loads, rates,
            state_trajectory, mean_donor,
            mean_acceptor,
            temperature)

        loads, state_trajectory =
            sample_loads_state_trajectory(draw, loads, rates,
                state_trajectory, mean_donor,
                mean_acceptor,
                temperature)

        mcmc_save_loads[draw, 1:max_n_system_states] = loads[:]
        mcmc_save_rates[draw, 1:n_rates] = rates[:]
        mcmc_log_posterior[draw] =
            get_log_posterior(loads, rates, mean_donor,
                mean_acceptor)
        mcmc_save_trajectory[draw, :] = state_trajectory[:]
        mcmc_save_mean_donor[draw, :] = mean_donor
        mcmc_save_mean_acceptor[draw, :] = mean_acceptor

        if isnan(mcmc_log_posterior[draw]) == false &&
           (maximum(filter(x -> isnan(x) != true,
            mcmc_log_posterior[1:draw])) == mcmc_log_posterior[draw])

            mcmc_save_MAP_trajectory = copy(state_trajectory)
        end

        generator, rho = get_generator(loads, rates)
        FRET_efficiencies = get_FRET_efficiencies(loads, rates)
        loads_active = filter(x -> x != 0, loads)
        n_system_states = size(loads_active)[1]

        flush(stdout)

        save_mcmc_data(draw, mcmc_save_loads, mcmc_save_rates,
            mcmc_log_posterior, mcmc_save_trajectory,
            mcmc_save_MAP_trajectory, mcmc_save_mean_donor,
            mcmc_save_mean_acceptor)

        if draw % 100 == 0
            println("****************************************************************")
            @show draw

            @show loads, n_system_states
            @show temperature
            @show generator
            @show FRET_efficiencies
            @show mean_donor[loads_active]
            @show mean_acceptor[loads_active]
            @show mcmc_log_posterior[draw]

            plot_everything(draw, state_trajectory, max_n_system_states,
                mcmc_log_posterior)
        end
        # Garbage Collection and free memory
        GC.gc()
    end

    println(" Done.")
    flush(stdout)

    return nothing
end

total_draws::Int64 = 30000
sampler_HMM(total_draws)
