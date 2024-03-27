function get_data()
    file_name = string(working_directory, file_prefix, ".h5")
    fid = h5open(file_name, "r")
    donor_channel_data = read(fid, "donor_channel")
    acceptor_channel_data = read(fid, "acceptor_channel")
    donor_channel_bg = read(fid, "donor_channel_bg")
    acceptor_channel_bg = read(fid, "acceptor_channel_bg")
    nbins = size(donor_channel_data)[1]
    close(fid)

    return donor_channel_data, acceptor_channel_data,
    donor_channel_bg, acceptor_channel_bg, nbins
end

function get_gt_trajectory()
    file_name = string(working_directory, "gt_", file_prefix, ".h5")
    fid = h5open(file_name, "r")
    binned_trajectory = read(fid, "binned_trajectory")
    close(fid)

    return binned_trajectory
end


function initialize_variables()

    # Initialize Rates

    # Rates
    # param = 1 (absorption rate/excitation rate): lambda_abs
    # param = 2 (excitation gradient)
    # param = 2:1+n_system_states (absorption rate/excitation rate): FRET_eff
    rates::Vector{Float64} = zeros(max_n_system_states^2)
    for i in 1:max_n_system_states
        for j in 1:max_n_system_states
            ij = (i - 1) * max_n_system_states + j
            if i == j # FRET Eff
                rates[ij] = rand()
            elseif i != j
                rates[ij] = rand(Gamma(1.0, 1.0), 1)[1] # s^-1
            end
        end
    end

    # Initialize loads for nonparametrics
    loads = zeros(Int64, max_n_system_states)
    if modeling_choice == "nonparametric"
        prior_success_probability::Float64 =
            1.0 / (1.0 + ((max_n_system_states - 1) /
                          expected_n_system_states))
        p_load::Vector{Float64} = [prior_success_probability,
            1.0 - prior_success_probability]
        n_system_states::Int64 = 0
        for i in 1:max_n_system_states
            loads[i] = rand(Categorical(p_load), 1)[1]
            if loads[i] == 1 #Active
                loads[i] = i
                n_system_states = n_system_states + 1
            elseif loads[i] == 2 #Inactive
                loads[i] = 0
                if i == max_n_system_states && n_system_states == 0
                    loads[i] = i
                end
            end
        end
    else
        for i in 1:expected_n_system_states
            loads[i] = i
        end
        n_system_states = expected_n_system_states
    end

    loads_active::Vector{Int64} = filter(x -> x != 0, loads)
    n_system_states = size(loads_active)[1]
    generator::Matrix{Float64}, rho::Vector{Float64} =
        get_generator(loads, rates)
    propagator::Matrix{Float64} = exp(bin_width .* generator)

    # Emission Parameters
    mean_donor::Vector{Float64} = zeros(max_n_system_states)
    variance_donor::Vector{Float64} = zeros(max_n_system_states)
    mean_acceptor::Vector{Float64} = zeros(max_n_system_states)
    variance_acceptor::Vector{Float64} = zeros(max_n_system_states)

    mean_donor .= sum(donor_channel_data) / size(donor_channel_data)[1]
    variance_donor .= sum((donor_channel_data .- mean_donor[1]) .^ 2) /
                      size(donor_channel_data)[1]
    mean_acceptor .= sum(acceptor_channel_data) / size(acceptor_channel_data)[1]
    variance_acceptor .= sum((acceptor_channel_data .- mean_acceptor[1]) .^ 2) /
                         size(acceptor_channel_data)[1]


    state_trajectory = ones(Int64, nbins)
    #	state_trajectory = get_gt_trajectory()
    previous_state::Int64 = rand(Categorical(rho), 1)[1]

    local next_state::Int64

    for bin in 1:nbins

        next_state =
            rand(Categorical(propagator[previous_state, :]), 1)[1]
        state_trajectory[bin] = loads_active[next_state]
        previous_state = next_state
        FRET_eff = rates[(loads_active[next_state]-1)*max_n_system_states+loads_active[next_state]]
    end

    return loads, rates, state_trajectory,
    mean_donor, variance_donor, mean_acceptor, variance_acceptor
end


# Get filter terms
function get_filter_terms(loads::Vector{Int64},
    rates::Vector{Float64},
    state_trajectory::Vector{Int64},
    mean_donor::Vector{Float64}, variance_donor::Vector{Float64},
    mean_acceptor::Vector{Float64}, variance_acceptor::Vector{Float64})

    generator::Matrix{Float64}, rho::Vector{Float64} =
        get_generator(loads, rates)
    propagator::Matrix{Float64} = exp(bin_width .* generator)
    loads_active::Vector{Int64} = filter(x -> x != 0, loads)
    n_system_states::Int64 = size(loads_active)[1]

    filter_terms::Matrix{Float64} = zeros(n_system_states, nbins)
    log_observation_prob::Vector{Float64} = zeros(n_system_states)
    prob_vec::Vector{Float64} = copy(rho)

    bin::Int64 = 1
    for system_state in 1:n_system_states
        log_observation_prob[system_state] =
            get_log_observation_prob(bin, loads_active[system_state],
                rates, mean_donor, variance_donor,
                mean_acceptor, variance_acceptor)
    end

    log_prob_max::Float64 = maximum(log_observation_prob)
    log_observation_prob_rescaled::Vector{Float64} =
        log_observation_prob .- log_prob_max
    observation_prob::Vector{Float64} = exp.(log_observation_prob_rescaled)
    prob_vec = prob_vec .* observation_prob
    prob_vec = prob_vec / (sum(prob_vec))
    filter_terms[1:n_system_states, bin] = prob_vec[1:n_system_states]
    local accept_trajectory::Bool

    if size(findall(x -> isnan(x) == true, prob_vec))[1] > 0 ||
       log_observation_prob == NaN

        accept_trajectory = false
        return filter_terms, accept_trajectory

    end
    for bin in 2:nbins
        prob_vec = permutedims(propagator) * prob_vec
        for system_state in 1:n_system_states
            log_observation_prob[system_state] =
                get_log_observation_prob(bin, loads_active[system_state],
                    rates, mean_donor, variance_donor,
                    mean_acceptor, variance_acceptor)
        end
        log_prob_max = maximum(log_observation_prob)
        log_observation_prob_rescaled = log_observation_prob .- log_prob_max

        observation_prob = exp.(log_observation_prob_rescaled)
        prob_vec = prob_vec .* observation_prob
        prob_vec = prob_vec / sum(prob_vec)
        filter_terms[1:n_system_states, bin] = prob_vec[1:n_system_states]


        if size(findall(x -> isnan(x) == true, prob_vec))[1] > 0 ||
           log_observation_prob == NaN

            accept_trajectory = false
            return filter_terms, accept_trajectory

        end
    end
    accept_trajectory = true

    return filter_terms, accept_trajectory
end

# State Trajectory
function sample_state_trajectory(loads::Vector{Int64},
    rates::Vector{Float64},
    mean_donor::Vector{Float64}, variance_donor::Vector{Float64},
    mean_acceptor::Vector{Float64}, variance_acceptor::Vector{Float64},
    temperature::Float64)

    generator::Matrix{Float64}, rho::Vector{Float64} =
        get_generator(loads, rates)
    propagator::Matrix{Float64} = exp(bin_width .* generator)
    state_trajectory = zeros(Int64, nbins)

    loads_active::Vector{Int64} = filter(x -> x != 0, loads)
    n_system_states::Int64 = size(loads_active)[1]

    # Get Filter Terms for State Trajectory
    filter_terms::Matrix{Float64}, accept_trajectory::Bool =
        get_filter_terms(loads, rates, state_trajectory,
            mean_donor, variance_donor,
            mean_acceptor, variance_acceptor)

    if accept_trajectory == true
        #Backward Sampling
        state_trajectory[nbins] = loads_active[rand(Categorical(
                filter_terms[1:n_system_states, nbins]), 1)[1]]
        for bin in nbins-1:-1:1
            final_state =
                findall(x -> x == state_trajectory[bin+1], loads_active)[1]
            prob_vec = filter_terms[1:n_system_states, bin] .*
                       propagator[1:n_system_states, final_state]
            prob_vec = prob_vec / sum(prob_vec)
            state_trajectory[bin] =
                loads_active[rand(Categorical(prob_vec), 1)[1]]
        end
    end

    return state_trajectory, accept_trajectory
end

function get_log_demarginalized_likelihood(loads::Vector{Int64},
    rates::Vector{Float64},
    state_trajectory::Vector{Int64},
    mean_donor::Vector{Float64}, variance_donor::Vector{Float64},
    mean_acceptor::Vector{Float64}, variance_acceptor::Vector{Float64})

    loads_active::Vector{Int64} = filter(x -> x != 0, loads)
    n_system_states::Int64 = size(loads_active)[1]
    generator::Matrix{Float64}, rho::Vector{Float64} =
        get_generator(loads, rates)
    propagator::Matrix{Float64} = propagator = exp(bin_width .* generator)
    log_demarginalized_likelihood::Float64 = 0.0

    local initial_state::Int64
    local final_state::Int64
    if n_system_states > 1
        for i in 1:n_system_states
            initial_state = i
            final_state = findall(x -> x == state_trajectory[1], loads_active)[1]
            log_demarginalized_likelihood += log(rho[initial_state]) +
                                             log(propagator[initial_state, final_state])
        end
        for bin in 2:nbins
            initial_state =
                findall(x -> x == state_trajectory[bin-1], loads_active)[1]
            final_state =
                findall(x -> x == state_trajectory[bin], loads_active)[1]
            log_demarginalized_likelihood += log(propagator[initial_state,
                final_state])
        end
    end
    for bin in 1:nbins
        log_demarginalized_likelihood +=
            get_log_observation_prob(bin, state_trajectory[bin],
                rates, mean_donor, variance_donor,
                mean_acceptor, variance_acceptor)
    end

    return log_demarginalized_likelihood
end

# State Trajectory
function sample_loads_state_trajectory(draw::Int64,
    all_loads::Vector{Int64},
    rates::Vector{Float64},
    state_trajectory::Vector{Int64},
    mean_donor::Vector{Float64}, variance_donor::Vector{Float64},
    mean_acceptor::Vector{Float64}, variance_acceptor::Vector{Float64},
    temperature::Float64)

    loads::Vector{Int64} = deepcopy(all_loads)
    old_state_trajectory::Vector{Int64} = copy(state_trajectory)
    old_log_demarginalized_likelihood::Float64 =
        get_log_demarginalized_likelihood(loads,
            rates, old_state_trajectory,
            mean_donor, variance_donor,
            mean_acceptor, variance_acceptor)

    # Sample loads
    prior_success_probability::Float64 =
        1.0 / (1.0 + ((max_n_system_states - 1) /
                      expected_n_system_states))

    local proposed_loads::Vector{Int64}
    local state_trajectory_inactive::Vector{Int64}
    local state_trajectory_active::Vector{Int64}
    local log_likelihood_inactive::Float64
    local log_likelihood_active::Float64
    local log_p_inactive::Float64
    local log_p_active::Float64
    local accept_trajectory::Bool

    sampled_state_trajectory, accept_trajectory =
        sample_state_trajectory(loads, rates,
            mean_donor, variance_donor,
            mean_acceptor, variance_acceptor,
            temperature)


    #	for i in 1:max_n_system_states
    # 		# Initialize proposed loads array
    #		proposed_loads = copy(loads)
    #		if proposed_loads[i] == 0
    #
    #			state_trajectory_inactive = copy(old_state_trajectory)
    #			log_likelihood_inactive = old_log_demarginalized_likelihood
    #			log_p_inactive = 1.0/temperature*log_likelihood_inactive + log(1.0 -
    #									prior_success_probability)
    #			proposed_loads[i] = i
    #			n_system_states = size(filter(x-> x != 0,
    #									proposed_loads))[1]
    #		    state_trajectory_active, accept_trajectory =
    #						sample_state_trajectory(proposed_loads, rates,
    #							mean_donor, variance_donor,
    #							mean_acceptor, variance_acceptor,
    #							temperature)
    #			if accept_trajectory == true
    #				log_likelihood_active =
    #					get_log_demarginalized_likelihood(proposed_loads,
    #							rates, state_trajectory_active,
    #							mean_donor, variance_donor,
    #							mean_acceptor, variance_acceptor)
    #				log_p_active = 1.0/temperature*log_likelihood_active +
    #									log(prior_success_probability)
    #			elseif accept_trajectory == false
    #
    #				log_p_active = -Inf
    #
    #			end
    #
    #		elseif proposed_loads[i] == i
    #
    #			state_trajectory_active = copy(old_state_trajectory)
    #			log_likelihood_active = old_log_demarginalized_likelihood
    #			log_p_active = 1.0/temperature*log_likelihood_active +
    #								log(prior_success_probability)
    # 			# log-Posterior associated with old parameters
    #			proposed_loads[i] = 0
    #			n_system_states = size(filter(x-> x != 0,
    #									proposed_loads))[1]
    #
    #			if n_system_states > 0
    #
    #		    	state_trajectory_inactive, accept_trajectory =
    #						sample_state_trajectory(proposed_loads, rates,
    #							mean_donor, variance_donor,
    #							mean_acceptor, variance_acceptor,
    #							temperature)
    #
    #				if accept_trajectory == true
    #					log_likelihood_inactive =
    #						get_log_demarginalized_likelihood(
    #							proposed_loads,
    #							rates, state_trajectory_inactive,
    #							mean_donor, variance_donor,
    #							mean_acceptor, variance_acceptor)
    # 					log_p_inactive = 1.0/temperature*log_likelihood_inactive +
    #									log(1.0 - prior_success_probability)
    #				elseif accept_trajectory == false
    #					log_p_inactive = -Inf
    #				end
    #
    #			elseif n_system_states == 0
    #				# Posterior is 0 for a zero-state model. So logarithm
    #				# of posterior is -Inf
    #				#
    #				log_p_inactive = -Inf
    #			end
    #
    #		end
    #
    #		# The following procedure helps avoid overflow issues
    #		if log_p_inactive == -Inf && log_p_active == -Inf
    #			println("trajectories problematic")
    #		else
    #			max_val = max(log_p_active, log_p_inactive)
    #			p_active = exp(log_p_active-max_val)
    #			p_inactive = exp(log_p_inactive-max_val)
    #
    #			# Probability vector to activate or deactivate a load
    #			p_load = [p_active, p_inactive]
    #
    #			# Normalize this probability vector
    #			p_load = p_load ./ sum(p_load)
    #			loads[i] =  rand(Categorical(p_load), 1)[1]
    #			if loads[i] == 1 #Active load
    #				loads[i] = i
    #				old_log_demarginalized_likelihood = log_likelihood_active
    #				old_state_trajectory = state_trajectory_active
    #			elseif loads[i] == 2 #Inactive load
    #				loads[i] = 0
    #				old_log_demarginalized_likelihood = log_likelihood_inactive
    #				old_state_trajectory = state_trajectory_inactive
    #			end
    #		end
    #	end

    #	return loads, old_state_trajectory
    return loads, sampled_state_trajectory

end

#function sample_rho(state_trajectory)
#
#	rho = zeros(n_system_states)
#	# Sample Propbabilities
#	alpha = ones(Int64, n_system_states)
#	alpha[state_trajectory[1]] += 1
#
# 	probabilities[1:n_states] = vec(rand(Dirichlet(alpha), 1))
#
#	return rho
#end

function get_log_likelihood_trajectory(loads::Vector{Int64},
    rates::Vector{Float64}, state_trajectory::Vector{Int64})

    loads_active::Vector{Int64} = filter(x -> x != 0, loads)
    n_system_states::Int64 = size(loads_active)[1]
    generator::Matrix{Float64}, rho::Vector{Float64} =
        get_generator(loads, rates)
    propagator::Matrix{Float64} = exp(bin_width .* generator)

    log_likelihood::Float64 = 0.0

    local initial_state::Int64
    local final_state::Int64
    if n_system_states > 1
        for i in 1:n_system_states
            initial_state = i
            final_state = findall(x -> x == state_trajectory[1], loads_active)[1]
            log_likelihood += log(rho[initial_state]) +
                              log(propagator[initial_state, final_state])
        end
        for bin in 2:nbins
            initial_state = findall(x -> x ==
                                         state_trajectory[bin-1], loads_active)[1]
            final_state = findall(x -> x ==
                                       state_trajectory[bin], loads_active)[1]
            log_likelihood += log(propagator[initial_state,
                final_state])
        end
    end

    return log_likelihood
end

function sample_transition_rates(draw::Int64,
    loads::Vector{Int64},
    rates::Vector{Float64},
    state_trajectory::Vector{Int64},
    mean_donor::Vector{Float64}, variance_donor::Vector{Float64},
    mean_acceptor::Vector{Float64}, variance_acceptor::Vector{Float64},
    temperature::Float64)

    loads_active::Vector{Int64} = filter(x -> x != 0, loads)
    loads_inactive::Vector{Int64} = findall(x -> x == 0, loads)

    n_system_states::Int64 = size(loads_active)[1]
    generator::Matrix{Float64}, rho::Vector{Float64} =
        get_generator(loads, rates)
    propagator::Matrix{Float64} = exp(bin_width .* generator)

    # For absorption rate
    old_log_conditional_likelihood::Float64 =
        get_log_demarginalized_likelihood(loads,
            rates, state_trajectory,
            mean_donor, variance_donor,
            mean_acceptor, variance_acceptor)


    std_rates::Float64 = 1.0
    conc_parameter_FRET::Float64 = 1.0e4

    local old_log_prior::Float64
    local old_log_conditional_posterior::Float64
    local new_log_conditional_likelihood::Float64
    local new_log_prior::Float64
    local new_log_conditional_posterior::Float64
    local log_hastings::Float64
    local proposed_rates::Vector{Float64}

    for i in loads_active
        for j in loads_active

            ij = (i - 1) * max_n_system_states + j
            if i != j  # For Confo Rates
                old_log_prior = logpdf(Gamma(1.0, 1.0), rates[ij])
                old_log_conditional_posterior =
                    old_log_conditional_likelihood + old_log_prior

                proposed_rates = copy(rates)
                proposed_rates[ij] =
                    rand(Normal(log(rates[ij]), std_rates), 1)[1]
                proposed_rates[ij] = exp(proposed_rates[ij])

                new_log_conditional_likelihood =
                    get_log_demarginalized_likelihood(loads,
                        proposed_rates, state_trajectory,
                        mean_donor, variance_donor,
                        mean_acceptor, variance_acceptor)
                new_log_prior = logpdf(Gamma(1.0, 1.0), proposed_rates[ij])
                new_log_conditional_posterior =
                    new_log_conditional_likelihood + new_log_prior

                log_hastings = (new_log_conditional_posterior -
                                old_log_conditional_posterior) +
                               log(proposed_rates[ij]) - log(rates[ij])

                if log_hastings >= log(rand())
                    old_log_conditional_likelihood =
                        new_log_conditional_likelihood
                    rates = copy(proposed_rates)
                end

            elseif i == j # For FRET Eff

                old_log_prior = logpdf(Dirichlet(ones(2)),
                    [rates[ij] + 1.0e-10, 1.0 - rates[ij] + 1.0e-10])
                old_log_conditional_posterior =
                    old_log_conditional_likelihood + old_log_prior

                proposed_rates = copy(rates)
                proposed_rates[ij] =
                    rand(Dirichlet(conc_parameter_FRET .* [rates[ij] + 1.0e-10,
                            1.0 - rates[ij] + 1.0e-10]), 1)[1]

                new_log_conditional_likelihood =
                    get_log_demarginalized_likelihood(loads,
                        proposed_rates, state_trajectory,
                        mean_donor, variance_donor,
                        mean_acceptor, variance_acceptor)

                new_log_prior = logpdf(Dirichlet(ones(2)),
                    [proposed_rates[ij] + 1.0e-10,
                        1.0 - proposed_rates[ij] + 1.0e-10])
                new_log_conditional_posterior =
                    new_log_conditional_likelihood + new_log_prior
                log_hastings =
                    new_log_conditional_posterior -
                    old_log_conditional_posterior +
                    logpdf(Dirichlet(conc_parameter_FRET .*
                                     [proposed_rates[ij] + 1.0e-10,
                            1.0 - proposed_rates[ij] + 1.0e-10]),
                        [rates[ij], 1.0 - rates[ij]]) -
                    logpdf(Dirichlet(conc_parameter_FRET .*
                                     [rates[ij] + 1.0e-10, 1.0 - rates[ij] + 1.0e-10]),
                        [proposed_rates[ij] + 1.0e-10,
                            1.0 - proposed_rates[ij] + 1.0e-10])

                if log_hastings >= log(rand())
                    old_log_conditional_likelihood =
                        new_log_conditional_likelihood
                    rates = copy(proposed_rates)
                end

            end
        end
    end

    for i in loads_inactive
        for j in loads_inactive

            ij = (i - 1) * max_n_system_states + j
            if i != j# For Confo Rates
                old_log_prior = logpdf(Gamma(1.0, 1.0), rates[ij])
                old_log_conditional_posterior = old_log_prior

                proposed_rates = copy(rates)
                proposed_rates[ij] = rand(Normal(log(rates[ij]),
                        std_rates), 1)[1]
                proposed_rates[ij] = exp(proposed_rates[ij])

                new_log_prior = logpdf(Gamma(1.0, 1.0), proposed_rates[ij])
                new_log_conditional_posterior = new_log_prior

                log_hastings = (new_log_conditional_posterior -
                                old_log_conditional_posterior) +
                               log(proposed_rates[ij]) - log(rates[ij])

                if log_hastings >= log(rand())
                    rates = copy(proposed_rates)
                end
            elseif i == j # For FRET_eff

                old_log_prior = logpdf(Dirichlet(ones(2)),
                    [rates[ij] + 1.0e-10, 1.0 - rates[ij] + 1.0e-10])
                old_log_conditional_posterior = old_log_prior

                proposed_rates = copy(rates)
                proposed_rates[ij] =
                    rand(Dirichlet(conc_parameter_FRET .* [rates[ij] + 1.0e-10,
                            1.0 - rates[ij] + 1.0e-10]), 1)[1]

                new_log_prior = logpdf(Dirichlet(ones(2)),
                    [proposed_rates[ij] + 1.0e-10,
                        1.0 - proposed_rates[ij] + 1.0e-10])
                new_log_conditional_posterior = new_log_prior
                log_hastings =
                    new_log_conditional_posterior -
                    old_log_conditional_posterior +
                    logpdf(Dirichlet(conc_parameter_FRET .*
                                     [proposed_rates[ij] + 1.0e-10,
                            1.0 - proposed_rates[ij] + 1.0e-10]),
                        [rates[ij] + 1.0e-10, 1.0 - rates[ij] + 1.0e-10]) -
                    logpdf(Dirichlet(conc_parameter_FRET .*
                                     [rates[ij] + 1.0e-10, 1.0 - rates[ij] + 1.0e-10]),
                        [proposed_rates[ij] + 1.0e-10,
                            1.0 - proposed_rates[ij] + 1.0e-10])

                if log_hastings >= log(rand())
                    rates = copy(proposed_rates)
                end
            end
        end
    end

    return rates
end

function get_log_likelihood_observations_only(rates::Vector{Float64},
    state_trajectory::Vector{Int64},
    mean_donor::Vector{Float64}, variance_donor::Vector{Float64},
    mean_acceptor::Vector{Float64}, variance_acceptor::Vector{Float64})

    log_likelihood_obs_only::Float64 = 0.0
    for bin in 1:nbins
        log_likelihood_obs_only +=
            get_log_observation_prob(bin, state_trajectory[bin],
                rates, mean_donor, variance_donor,
                mean_acceptor, variance_acceptor)
    end
    return log_likelihood_obs_only
end

function sample_emission_parameters(draw::Int64,
    loads::Vector{Int64},
    rates::Vector{Float64},
    state_trajectory::Vector{Int64},
    mean_donor::Vector{Float64}, variance_donor::Vector{Float64},
    mean_acceptor::Vector{Float64}, variance_acceptor::Vector{Float64},
    temperature::Float64)

    loads_active::Vector{Int64} = filter(x -> x != 0, loads)
    loads_inactive::Vector{Int64} = findall(x -> x == 0, loads)
    n_system_states::Int64 = size(loads_active)[1]

    old_log_conditional_likelihood::Float64 =
        get_log_likelihood_observations_only(rates,
            state_trajectory, mean_donor, variance_donor,
            mean_acceptor, variance_acceptor)

    std_mean = 5.0
    std_variance = 10.0

    for i in 1:max_n_system_states

        if loads[i] != 0

            # For mean_donor
            old_log_prior = logpdf(Normal(1000.0, 100000.0), mean_donor[i])
            old_log_conditional_posterior =
                old_log_conditional_likelihood + old_log_prior


            proposed_mean_donor = copy(mean_donor)
            proposed_mean_donor[i] = rand(Normal(mean_donor[i], std_mean), 1)[1]

            if proposed_mean_donor[i] > 0.0
                new_log_conditional_likelihood =
                    get_log_likelihood_observations_only(rates,
                        state_trajectory, proposed_mean_donor,
                        variance_donor,
                        mean_acceptor, variance_acceptor)

                new_log_prior = logpdf(Normal(1000.0, 100000.0), proposed_mean_donor[i])
                new_log_conditional_posterior =
                    new_log_conditional_likelihood + new_log_prior
                log_hastings =
                    new_log_conditional_posterior - old_log_conditional_posterior
                if log_hastings >= log(rand())
                    old_log_conditional_likelihood = new_log_conditional_likelihood
                    mean_donor = copy(proposed_mean_donor)
                end
            end

            # For variance_donor
            old_log_prior = logpdf(Normal(100000.0, 100000.0), variance_donor[i])
            old_log_conditional_posterior =
                old_log_conditional_likelihood + old_log_prior


            proposed_variance_donor = copy(variance_donor)
            proposed_variance_donor[i] =
                rand(Normal(variance_donor[i], std_variance), 1)[1]

            if proposed_variance_donor[i] > 0.0
                new_log_conditional_likelihood =
                    get_log_likelihood_observations_only(rates,
                        state_trajectory, mean_donor,
                        proposed_variance_donor,
                        mean_acceptor, variance_acceptor)

                new_log_prior = logpdf(Normal(100000.0, 100000.0), proposed_variance_donor[i])
                new_log_conditional_posterior =
                    new_log_conditional_likelihood + new_log_prior
                log_hastings =
                    new_log_conditional_posterior - old_log_conditional_posterior
                if log_hastings >= log(rand())
                    old_log_conditional_likelihood = new_log_conditional_likelihood
                    variance_donor = copy(proposed_variance_donor)
                end
            end
            #

            # For mean_acceptor
            old_log_prior = logpdf(Normal(1000.0, 100000.0), mean_acceptor[i])
            old_log_conditional_posterior =
                old_log_conditional_likelihood + old_log_prior


            proposed_mean_acceptor = copy(mean_acceptor)
            proposed_mean_acceptor[i] =
                rand(Normal(mean_acceptor[i], std_mean), 1)[1]

            if proposed_mean_acceptor[i] > 0.0
                new_log_conditional_likelihood =
                    get_log_likelihood_observations_only(rates,
                        state_trajectory, mean_donor, variance_donor,
                        proposed_mean_acceptor, variance_acceptor)

                new_log_prior = logpdf(Normal(1000.0, 100000.0), proposed_mean_acceptor[i])
                new_log_conditional_posterior =
                    new_log_conditional_likelihood + new_log_prior
                log_hastings =
                    new_log_conditional_posterior - old_log_conditional_posterior
                if log_hastings >= log(rand())
                    old_log_conditional_likelihood = new_log_conditional_likelihood
                    mean_acceptor = copy(proposed_mean_acceptor)
                end
            end

            # For variance_acceptor
            old_log_prior = logpdf(Normal(100000.0, 100000.0), variance_acceptor[i])
            old_log_conditional_posterior =
                old_log_conditional_likelihood + old_log_prior


            proposed_variance_acceptor = copy(variance_acceptor)
            proposed_variance_acceptor[i] =
                rand(Normal(variance_acceptor[i], std_variance), 1)[1]

            if proposed_variance_acceptor[i] > 0.0
                new_log_conditional_likelihood =
                    get_log_likelihood_observations_only(rates,
                        state_trajectory, mean_donor, variance_donor,
                        mean_acceptor, proposed_variance_acceptor)

                new_log_prior =
                    logpdf(Normal(100000.0, 100000.0), proposed_variance_acceptor[i])
                new_log_conditional_posterior =
                    new_log_conditional_likelihood + new_log_prior
                log_hastings =
                    new_log_conditional_posterior - old_log_conditional_posterior
                if log_hastings >= log(rand())
                    old_log_conditional_likelihood = new_log_conditional_likelihood
                    variance_acceptor = copy(proposed_variance_acceptor)
                end
            end

        elseif loads[i] == 0

            # For mean_donor
            old_log_prior = logpdf(Normal(1000.0, 100000.0), mean_donor[i])
            old_log_conditional_posterior = old_log_prior

            proposed_mean_donor = copy(mean_donor)
            proposed_mean_donor[i] = rand(Normal(mean_donor[i],
                    std_mean), 1)[1]

            if proposed_mean_donor[i] > 0.0
                new_log_prior = logpdf(Normal(1000.0, 100000.0), proposed_mean_donor[i])
                new_log_conditional_posterior = new_log_prior
                log_hastings =
                    new_log_conditional_posterior - old_log_conditional_posterior
                if log_hastings >= log(rand())
                    mean_donor = copy(proposed_mean_donor)
                end
            end

            # For variance_donor
            old_log_prior = logpdf(Normal(100000.0, 100000.0), variance_donor[i])
            old_log_conditional_posterior = old_log_prior

            proposed_variance_donor = copy(variance_donor)
            proposed_variance_donor[i] =
                rand(Normal(variance_donor[i], std_variance), 1)[1]

            if proposed_variance_donor[i] > 0.0
                new_log_prior = logpdf(Normal(100000.0, 100000.0),
                    proposed_variance_donor[i])
                new_log_conditional_posterior = new_log_prior
                log_hastings =
                    new_log_conditional_posterior - old_log_conditional_posterior
                if log_hastings >= log(rand())
                    variance_donor = copy(proposed_variance_donor)
                end
            end
            #

            # For mean_acceptor
            old_log_prior = logpdf(Normal(1000.0, 100000.0), mean_acceptor[i])
            old_log_conditional_posterior = old_log_prior


            proposed_mean_acceptor = copy(mean_acceptor)
            proposed_mean_acceptor[i] =
                rand(Normal(mean_acceptor[i], std_mean), 1)[1]

            if proposed_mean_acceptor[i] > 0.0
                new_log_prior = logpdf(Normal(1000.0, 100000.0),
                    proposed_mean_acceptor[i])
                new_log_conditional_posterior = new_log_prior
                log_hastings =
                    new_log_conditional_posterior - old_log_conditional_posterior
                if log_hastings >= log(rand())
                    mean_acceptor = copy(proposed_mean_acceptor)
                end
            end

            # For variance_acceptor
            old_log_prior = logpdf(Normal(100000.0, 100000.0),
                variance_acceptor[i])
            old_log_conditional_posterior = old_log_prior

            proposed_variance_acceptor = copy(variance_acceptor)
            proposed_variance_acceptor[i] =
                rand(Normal(variance_acceptor[i],
                        std_variance), 1)[1]
            if proposed_variance_acceptor[i] > 0.0
                new_log_prior =
                    logpdf(Normal(100000.0, 100000.0),
                        proposed_variance_acceptor[i])
                new_log_conditional_posterior = new_log_prior
                log_hastings =
                    new_log_conditional_posterior - old_log_conditional_posterior
                if log_hastings >= log(rand())
                    variance_acceptor = copy(proposed_variance_acceptor)
                end
            end

        end
    end

    return rates, mean_donor, variance_donor,
    mean_acceptor, variance_acceptor
end #function


function get_reduced_propagator(bin::Int64,
    loads::Vector{Int64},
    rates::Vector{Float64},
    mean_donor::Vector{Float64}, variance_donor::Vector{Float64},
    mean_acceptor::Vector{Float64}, variance_acceptor::Vector{Float64})

    loads_active::Vector{Int64} = filter(x -> x != 0, loads)
    n_system_states::Int64 = size(loads_active)[1]
    generator::Matrix{Float64}, rho::Vector{Float64} =
        get_generator(loads, rates)
    propagator::Matrix{Float64} = exp(bin_width .* generator)

    reduced_propagator::Matrix{Float64} = copy(propagator)


    local obs_prob::Float64
    for state in 1:n_system_states
        obs_prob = exp(get_log_observation_prob(bin, loads_active[state],
            rates, mean_donor, variance_donor,
            mean_acceptor, variance_acceptor))
        reduced_propagator[state, :] = obs_prob * reduced_propagator[state, :]
    end

    return reduced_propagator
end

function get_log_posterior(loads::Vector{Int64},
    rates::Vector{Float64},
    mean_donor::Vector{Float64}, variance_donor::Vector{Float64},
    mean_acceptor::Vector{Float64}, variance_acceptor::Vector{Float64})

    loads_active::Vector{Int64} = filter(x -> x != 0, loads)
    n_system_states::Int64 = size(loads_active)[1]
    generator::Matrix{Float64}, rho::Vector{Float64} =
        get_generator(loads, rates)

    log_likelihood::Float64 = 0.0
    p_vec::Matrix{Float64} = permutedims(copy(rho))
    p::Float64 = 1.0

    local Q::Matrix{Float64}
    for bin in 1:nbins
        Q = get_reduced_propagator(bin, loads, rates,
            mean_donor, variance_donor,
            mean_acceptor, variance_acceptor)
        p_vec = p_vec * Q
        p = sum(p_vec)
        log_likelihood += log(p)
        p_vec = p_vec / p
    end

    log_prior::Float64 = 0.0
    prior_success_probability::Float64 = 1.0 / (1.0 + ((max_n_system_states - 1) /
                                                       expected_n_system_states))

    # Add Priors
    for i in 1:max_n_system_states
        # For loads
        if loads[i] == i
            log_prior += log(prior_success_probability)
        else
            log_prior += log(1.0 - prior_success_probability)
        end

        for j in 1:max_n_system_states
            ij = (i - 1) * max_n_system_states + j
            if i != j # Confo Rates
                log_prior += logpdf(Gamma(1.0, 1.0), rates[ij])
            elseif i == j # FRET Eff
                log_prior += logpdf(Dirichlet(ones(2)),
                    [rates[ij] + 1.0e-10, 1.0 - rates[ij] + 1.0e-10])
            end
        end
    end

    log_posterior::Float64 = log_likelihood + log_prior

    return log_posterior
end

function save_mcmc_data(current_draw::Int64,
    mcmc_save_loads::Matrix{Int64},
    mcmc_save_rates::Matrix{Float64},
    mcmc_log_posterior::Vector{Float64},
    mcmc_save_trajectory::Matrix{Int64},
    mcmc_save_MAP_trajectory::Vector{Int64},
    mcmc_save_mean_donor::Matrix{Float64}, mcmc_save_variance_donor::Matrix{Float64},
    mcmc_save_mean_acceptor::Matrix{Float64}, mcmc_save_variance_acceptor::Matrix{Float64})

    # Save the data in HDF5 format.
    file_name = string(working_directory, "mcmc_output_", file_prefix, ".h5")

    fid = h5open(file_name, "w")

    write_dataset(fid, "mcmc_loads",
        mcmc_save_loads[1:current_draw, :])
    write_dataset(fid, "mcmc_rates",
        mcmc_save_rates[1:current_draw, :])
    write_dataset(fid, "mcmc_log_posterior",
        mcmc_log_posterior[1:current_draw])
    write_dataset(fid, "mcmc_trajectory",
        mcmc_save_trajectory[1:current_draw, :])
    write_dataset(fid, "mcmc_trajectory_MAP", mcmc_save_MAP_trajectory[:])

    write_dataset(fid, "mcmc_mean_donor", mcmc_save_mean_donor[1:current_draw, :])
    write_dataset(fid, "mcmc_mean_acceptor", mcmc_save_mean_acceptor[1:current_draw, :])

    write_dataset(fid, "mcmc_variance_donor", mcmc_save_variance_donor[1:current_draw, :])
    write_dataset(fid, "mcmc_variance_acceptor", mcmc_save_variance_acceptor[1:current_draw, :])

    close(fid)

    return nothing
end

function check_existing_mcmc_data(mcmc_save_loads::Matrix{Int64},
    mcmc_save_rates::Matrix{Float64},
    mcmc_save_mean_donor::Matrix{Float64},
    mcmc_save_mean_acceptor::Matrix{Float64},
    mcmc_save_variance_donor::Matrix{Float64},
    mcmc_save_variance_acceptor::Matrix{Float64},
    mcmc_log_posterior::Vector{Float64},
    mcmc_save_trajectory::Matrix{Int64})

    # Save the data in HDF5 format.
    file_name = string(working_directory, "mcmc_output_", file_prefix, ".h5")

    local last_draw::Int64
    if isfile(file_name) == true

        fid = h5open(file_name, "r")
        old_mcmc_save_loads = read(fid, "mcmc_loads")
        old_mcmc_save_rates = read(fid, "mcmc_rates")
        old_mcmc_log_posterior = read(fid, "mcmc_log_posterior")
        old_mcmc_save_trajectory = read(fid, "mcmc_trajectory")
        old_mcmc_save_MAP_trajectory = read(fid, "mcmc_trajectory_MAP")

        old_mcmc_save_mean_donor = read(fid, "mcmc_mean_donor")
        old_mcmc_save_mean_acceptor = read(fid, "mcmc_mean_acceptor")

        old_mcmc_save_variance_donor = read(fid, "mcmc_variance_donor")
        old_mcmc_save_variance_acceptor = read(fid, "mcmc_variance_acceptor")

        close(fid)

        last_draw = size(old_mcmc_save_loads)[1]

        mcmc_save_loads[1:last_draw, :] = old_mcmc_save_loads[:, :]
        mcmc_save_rates[1:last_draw, :] = old_mcmc_save_rates[:, :]
        mcmc_save_mean_donor[1:last_draw, :] = old_mcmc_save_mean_donor[:, :]
        mcmc_save_mean_acceptor[1:last_draw, :] = old_mcmc_save_mean_acceptor[:, :]
        mcmc_save_variance_donor[1:last_draw, :] = old_mcmc_save_variance_donor[:, :]
        mcmc_save_variance_acceptor[1:last_draw, :] = old_mcmc_save_variance_acceptor[:, :]

        mcmc_log_posterior[1:last_draw] = old_mcmc_log_posterior[:]
        mcmc_save_trajectory[1:last_draw, :] =
            old_mcmc_save_trajectory[:, :]

    else
        last_draw = 1
        old_mcmc_save_MAP_trajectory = zeros(Int64, nbins)
    end

    return last_draw, old_mcmc_save_MAP_trajectory

end



function plot_everything(draw, step_found_trajectory, n_system_states,
    mcmc_log_posterior)

    nbins = size(donor_channel_data)[1]
    time_data = bin_width * collect(1:nbins)
    plot(time_data, (donor_channel_data) ./ 1000.0, color=:green,
        label=("Donor"), legendfontsize=14)
    plot_data = plot!(time_data, (acceptor_channel_data) ./ 1000.0,
        xlabel="Time (s)", ylabel="Intensity (a.u.)",
        legend=true, linewidth=1.5, label=("Acceptor"),
        #  xlims = (1, current_draw),
        #  xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
        #  yticks = (collect(1.0:convert(Float64, n_system_states)),
        # collect(1:n_system_states)),
        xtickfontsize=18, ytickfontsize=18,
        xguidefontsize=20, yguidefontsize=20,
        fontfamily="Computer Modern",
        right_margin=5mm, bottom_margin=5mm,
        left_margin=5mm, color=:red,
        dpi=600, format=:svg)
    plot_apparent = plot(time_data, (acceptor_channel_data) ./ (acceptor_channel_data +
                                                donor_channel_data),
        xlabel="Time (s)", ylabel="E\$_{FRET}\$",
        legend=false, linewidth=1.5,
        # xlims = (1, current_draw),
        # xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
        # yticks = (collect(1.0:convert(Float64, n_system_states)),
        # collect(1:n_system_states)),
        ylims=(0.0, 1.0),
        xtickfontsize=18, ytickfontsize=18,
        xguidefontsize=20, yguidefontsize=20,
        fontfamily="Computer Modern",
        right_margin=5mm, bottom_margin=5mm,
        left_margin=5mm, color=:blue,
        dpi=600, format=:svg)
    plot_stepdata = plot(time_data, step_found_trajectory,
        xlabel="Time (s)", ylabel="State",
        legend=false, linewidth=1.5,
        # xlims = (1, current_draw),
        # xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
        ylims=(0, n_system_states),
        yticks=(collect(0.0:1.0:convert(Float64, n_system_states)),
            collect(0:1:n_system_states)),
        xtickfontsize=18, ytickfontsize=18,
        xguidefontsize=20, yguidefontsize=20,
        fontfamily="Computer Modern",
        right_margin=5mm, bottom_margin=5mm,
        left_margin=5mm, color=:blue,
        dpi=600, format=:svg)

    plot_gt_trajectory = plot(time_data, gt_trajectory,
        xlabel="Time (s)", ylabel="State",
        legend=false, linewidth=1.5,
        # xlims = (1, current_draw),
        # xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
        ylims=(0, 2),
        yticks=(collect(0.0:1.0:convert(Float64, 2)),
            collect(0:1:2)),
        xtickfontsize=18, ytickfontsize=18,
        xguidefontsize=20, yguidefontsize=20,
        fontfamily="Computer Modern",
        right_margin=5mm, bottom_margin=5mm,
        left_margin=5mm, color=:blue,
        dpi=600, format=:svg)


    plot_posterior = plot(mcmc_log_posterior[1:draw],
        xlabel="Iterations", ylabel="log-posterior",
        legend=false, linewidth=1.5,
        # xlims = (1, current_draw),
        # xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
        #        ylims = (0,n_system_states),
        #        yticks = (collect(0.0:1.0:convert(Float64, n_system_states)),
        #			collect(0:1:n_system_states)),
        xtickfontsize=18, ytickfontsize=18,
        xguidefontsize=20, yguidefontsize=20,
        fontfamily="Computer Modern",
        right_margin=5mm, bottom_margin=5mm,
        left_margin=5mm, color=:blue,
        dpi=600, format=:svg)
    l = @layout [a; b; c; d; e]
    display(plot(plot_data, plot_apparent, plot_gt_trajectory,
        plot_stepdata, plot_posterior, layout=l, size=(1000, 1700), format=:svg))

    return nothing
end
