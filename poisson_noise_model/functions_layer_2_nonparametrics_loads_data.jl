# Get observation probability
function get_log_observation_prob(bin::Int64,
			state_label::Int64,
			rates::Vector{Float64},
			mean_donor::Vector{Float64},
			mean_acceptor::Vector{Float64})

	FRET_eff::Float64 = rates[(state_label - 1) * max_n_system_states +
									state_label]

	# Shot noise
	log_observation_prob::Float64 = logpdf(Poisson(
			mean_donor[state_label]),
					donor_channel_data[bin]) +
			logpdf(Poisson(
			mean_acceptor[state_label]),
					acceptor_channel_data[bin])


 	return log_observation_prob
end

function get_generator(loads::Vector{Int64}, rates::Vector{Float64})

	loads_active::Vector{Int64} = filter(x-> x != 0, loads)
	n_system_states::Int64 = size(loads_active)[1]
	rate_matrix::Matrix{Float64} = zeros(n_system_states, n_system_states)

	for i in 1:n_system_states
		for j in 1:n_system_states
			ij = (loads_active[i]-1)*max_n_system_states + loads_active[j]
			if i != j
				rate_matrix[i, j] = rates[ij]
			end
		end
	end
	generator::Matrix{Float64} = copy(rate_matrix)
	length_generator::Int64 = size(generator)[1]
	for i in 1:length_generator
		generator[i,i] = -sum(generator[i,:])
	end

	rho::Matrix{Float64} = nullspace(Transpose(generator))
	rho = rho/sum(rho)

	return generator, vec(rho)
end

function get_FRET_efficiencies(loads::Vector{Int64}, rates::Vector{Float64})

	loads_active::Vector{Int64} = filter(x-> x != 0, loads)
	n_system_states::Int64 = size(loads_active)[1]
	FRET_efficiencies::Vector{Float64} = zeros(n_system_states)

	for i in 1:n_system_states
			ii = (loads_active[i]-1)*max_n_system_states + loads_active[i]
				FRET_efficiencies[i] = rates[ii]
	end

	return FRET_efficiencies
end
