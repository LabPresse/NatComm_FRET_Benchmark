using Random, Distributions
using LinearAlgebra
using Statistics
using HDF5
using ProgressBars
using Plots, Plots.Measures

working_directory = "/home/singularity/Dropbox (ASU)/bayesianCodes/julia/multiFRET/BNP-FRET_binned/"
n_system_states = 2
FRET_eff = [0.38, 0.71]

crosstalk_da = 0.0
crosstalk_dd = 1.0 - crosstalk_da
crosstalk_ad = 0.0
crosstalk_aa = 1.0

dt = 0.05 # in seconds
#const quantum_yield_d::Float64 = 0.47
#const quantum_yield_a::Float64 = 0.47
#const bin_width::Float64 = 0.05 # in seconds
#const ccd_EM_gain_donor::Float64 = 40.0
#const ccd_EM_gain_acceptor::Float64 = 40.0
#const ccd_sensitivity_donor::Float64 = 4.97 # electrons per ADU
#const ccd_sensitivity_acceptor::Float64 = 4.97 # electrons per ADU
#const ccd_read_out_noise_donor::Float64 = 188
#const ccd_read_out_noise_acceptor::Float64 = 142
#const n_pixels_integrated_over::Int64 = 25
#const ccd_offset_donor::Float64 = n_pixels_integrated_over * 253.0
#const ccd_offset_acceptor::Float64 = n_pixels_integrated_over * 224.0
#const detection_eff_acceptor::Float64 = 1.0
#const da_crosstalk::Float64 = crosstalk_da

const quantum_yield_d::Float64 = 0.47
const quantum_yield_a::Float64 = 0.47
const bin_width::Float64 = 0.05 # in seconds
const ccd_EM_gain_donor::Float64 = 40.0
const ccd_EM_gain_acceptor::Float64 = 40.0
const ccd_sensitivity_donor::Float64 = 4.97 # electrons per ADU
const ccd_sensitivity_acceptor::Float64 = 4.97 # electrons per ADU
const ccd_read_out_noise_donor::Float64 = 188
const ccd_read_out_noise_acceptor::Float64 = 142
const n_pixels_integrated_over::Int64 = 25
const ccd_offset_donor::Float64 = n_pixels_integrated_over * 253.0
const ccd_offset_acceptor::Float64 = n_pixels_integrated_over * 224.0
.0
const detection_eff_acceptor::Float64 = 1.0
const da_crosstalk::Float64 = crosstalk_da

nbins= 2000
# Synthetic Data
function synthetic_data(draws)

    id2 = Matrix(1.0*I,2,2)
    id3 = Matrix(1.0*I,3,3)
    id4 = Matrix(1.0*I,4,4)
	idFRET = Matrix(1.0*I,n_system_states*3, n_system_states*3)

    matrix_confo = [0.0 1.5;
                    0.5 0.0;] # In s^-1
#    matrix_bg_1 = [0.0 0.0;
#	       0.0 0.0] # Acceptor Channel
#    matrix_bg_2 = [0.0 0.0;
#	       0.0 0.0] # Donor Channel
#    rate_matrix_FRET = kron(matrix_confo, id3)
#    for i in 1:n_system_states
#        FRET_rate = (FRET_eff[i]*(1.0/2.8))/
#                                (1.0-FRET_eff[i])
#        matrix_chromo = [0.0 1.0e-5 0.0;
#                         (1.0/3.5) 0.0 FRET_rate;
#                        (1.0/3.6) 0.0 0.0]
#        rate_matrix_FRET[(i-1)*3+1:(i-1)*3+3,
#                         (i-1)*3+1:(i-1)*3+3] =
#                            matrix_chromo[1:3, 1:3]
#    end
#
#	matrix_length_FRET = size(rate_matrix_FRET)[1]
#	generative_matrix_FRET = copy(rate_matrix_FRET)
#	for i in 1:matrix_length_FRET
# 		generative_matrix_FRET[i,i] = -sum(generative_matrix_FRET[i,:])
#	end
#
# 	rate_matrix_bg = kron(matrix_bg_1, id2) + kron(id2, matrix_bg_2)
#
#	matrix_length_bg = size(rate_matrix_bg)[1]
#	generative_matrix_bg = copy(rate_matrix_bg)
#	for i in 1:matrix_length_bg
# 		generative_matrix_bg[i,i] = -sum(generative_matrix_bg[i,:])
#	end

#	rate_matrix = kron(rate_matrix_bg, idFRET) + kron(id4, rate_matrix_FRET)
 	rate_matrix = copy(matrix_confo)

	# Sample the initial state
	p_initial = zeros(size(rate_matrix)[1])
	p_initial[1] = 1.0
	initial_state = rand(Categorical(p_initial),1)[1]

	# Initialize arrays and constants
	time_state = [0.0]
	state = [initial_state]
	time_photon = []
	color_photon = []

	#Sample the next state in two steps
   	hold_time = 0.0
   	next_event = 0.0
   	current_state = initial_state

	for i in ProgressBar(1:draws)

   		lambda_state = sum(rate_matrix[current_state, :])
   	   	p_next = (1.0/lambda_state) * rate_matrix[current_state, :]
   	   	next_state = rand(Categorical(p_next),1)[1]
   	   	hold_time = rand(Exponential(1.0/lambda_state),1)[1]
   	   	next_event = next_event + hold_time

   	   	time_state = vcat(time_state, next_event)
   	   	state = vcat(state, next_state)

   	   	current_state = next_state

	end


	return time_state, state
end


excitation_rate = 2000
for i in 41:160
	transition_time, system_state = synthetic_data(1000)



	binned_trajectory = zeros(Integer, nbins)
	photons_absorbed = zeros(Integer, nbins)
	emitted_donor_photons = zeros(Integer, nbins)
	emitted_acceptor_photons = zeros(Integer, nbins)
	multiplied_donor_electrons = zeros(Integer, nbins)
	multiplied_acceptor_electrons = zeros(Integer, nbins)
	donor_channel = zeros(Integer, nbins)
	acceptor_channel = zeros(Integer, nbins)


	bin = 1
	binned_trajectory[bin] = system_state[1]
	photons_absorbed[bin] = rand(Poisson(excitation_rate), 1)[1]
	emissions = rand(Multinomial(photons_absorbed[bin],
				[(1.0-FRET_eff[system_state[bin]]) * quantum_yield_d,
				FRET_eff[system_state[bin]] * quantum_yield_a,
				1.0 - (1.0-FRET_eff[system_state[bin]]) * quantum_yield_d -
				FRET_eff[system_state[bin]] * quantum_yield_a]))
	sum(emissions)

	emitted_donor_photons[bin] = emissions[1]
	emitted_acceptor_photons[bin] = emissions[2]

	multiplied_donor_electrons[bin] = Integer(
					round(rand(Gamma((1-da_crosstalk)*emitted_donor_photons[bin]/2,
					2*ccd_EM_gain_donor))))
	multiplied_acceptor_electrons[bin] = Integer(
				round(rand(Gamma((da_crosstalk*emitted_donor_photons[bin]+
				emitted_acceptor_photons[1])/2,
				2*ccd_EM_gain_acceptor))))

	donor_channel[bin] = Integer(
				round(rand(Normal(multiplied_donor_electrons[bin] *
				1.0/ccd_sensitivity_donor +
				ccd_offset_donor,
				ccd_read_out_noise_donor))))
	acceptor_channel[bin] = Integer(
				round(rand(Normal(multiplied_acceptor_electrons[bin] *
				1.0/ccd_sensitivity_acceptor +
				ccd_offset_acceptor,
				ccd_read_out_noise_acceptor))))

	for bin in 2:nbins

		min = transition_time ./ ((bin-1)*dt)
		min_first = findall(x -> x < 1.0, min)
		max = transition_time ./ (bin*dt)
		max_first = findall(x -> x < 1.0, max)

		binned_trajectory[bin] = system_state[min_first[end]]
		photons_absorbed[bin] = rand(Poisson(excitation_rate), 1)[1]

		emissions = rand(Multinomial(photons_absorbed[bin],
					[(1.0-FRET_eff[binned_trajectory[bin]]) * quantum_yield_d,
					FRET_eff[binned_trajectory[bin]] * quantum_yield_a,
					1.0 - (1.0-FRET_eff[binned_trajectory[bin]]) * quantum_yield_d -
					FRET_eff[binned_trajectory[bin]] * quantum_yield_a]))

		emitted_donor_photons[bin] = emissions[1]
		emitted_acceptor_photons[bin] = emissions[2]

		multiplied_donor_electrons[bin] = Integer(
					round(rand(Gamma((1-da_crosstalk)*emitted_donor_photons[bin]/2,
					2*ccd_EM_gain_donor))))
		multiplied_acceptor_electrons[bin] = Integer(
				round(rand(Gamma((da_crosstalk*emitted_donor_photons[bin]+
				emitted_acceptor_photons[bin])/2,
				2*ccd_EM_gain_acceptor))))

		donor_channel[bin] = Integer(
				round(rand(Normal(multiplied_donor_electrons[bin] *
				1.0/ccd_sensitivity_donor +
				ccd_offset_donor,
				ccd_read_out_noise_donor))))
		acceptor_channel[bin] = Integer(
				round(rand(Normal(multiplied_acceptor_electrons[bin] *
				1.0/ccd_sensitivity_acceptor +
				ccd_offset_acceptor,
				ccd_read_out_noise_acceptor))))

	end

	file_name = string(working_directory, "trace_true_ex_rate_2000_", i, ".h5")
	fid = h5open(file_name,"w")
	write_dataset(fid, "binned_trajectory", convert.(Int64, binned_trajectory))
	write_dataset(fid, "excitation_rate", convert.(Int64, excitation_rate))
	write_dataset(fid, "photons_absorbed", convert.(Int64, photons_absorbed))
	write_dataset(fid, "emitted_donor_photons", convert.(Int64, emitted_donor_photons))
	write_dataset(fid, "emitted_acceptor_photons", convert.(Int64, emitted_acceptor_photons))
	write_dataset(fid, "multiplied_donor_electrons", convert.(Int64, multiplied_donor_electrons))
	write_dataset(fid, "multiplied_acceptor_electrons", convert.(Int64, multiplied_acceptor_electrons))
	write_dataset(fid, "donor_channel", convert.(Int64, donor_channel))
	write_dataset(fid, "acceptor_channel", convert.(Int64, acceptor_channel))
	write_dataset(fid, "donor_channel_bg", ccd_offset_donor)
	write_dataset(fid, "acceptor_channel_bg", ccd_offset_acceptor)

	close(fid)
end



plot_donor = plot(dt .* collect(0:nbins-1), donor_channel,
	xlabel="Time (s)" , ylabel="Intensity (ADU)",
	legend = false, linewidth= 1.5,
	xlims = (-0.1, dt*nbins),
#	xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
#		ylims = (0.7,n_system_states),
# 	yticks = (collect(1.0:convert(Float64, n_system_states)),
# 	collect(1:n_system_states)),
	xtickfontsize=18, ytickfontsize=18,
	xguidefontsize = 20, yguidefontsize = 20,
	fontfamily = "Computer Modern",
	right_margin=5mm, bottom_margin = 12mm,
	left_margin = 10mm, color = :green,
	dpi = 600, format = :svg,
	size=(1200, 300))

plot_acceptor = plot!(dt .* collect(0:nbins-1), acceptor_channel,
	xlabel="Time (s)" , ylabel="Intensity (ADU)",
	legend = false, linewidth= 1.5,
	xlims = (-0.1, dt*nbins),
#	xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
#		ylims = (0.7,n_system_states),
# 	yticks = (collect(1.0:convert(Float64, n_system_states)),
# 	collect(1:n_system_states)),
	xtickfontsize=18, ytickfontsize=18,
	xguidefontsize = 20, yguidefontsize = 20,
	fontfamily = "Computer Modern",
	right_margin=5mm, bottom_margin = 12mm,
	left_margin = 10mm, color = :red,
	dpi = 600, format = :svg,
	size=(1200, 300))
plot!(binned_trajectory, line)
histogram(acceptor_channel./(donor_channel+acceptor_channel), bin = 60)












file_name = string(working_directory, "test_trace_true.h5")
fid = h5open(file_name,"w")
write_dataset(fid, "donor_channel", convert.(Int64, donor_channel))
write_dataset(fid, "acceptor_channel", convert.(Int64, acceptor_channel))
write_dataset(fid, "donor_channel_bg", ccd_offset_donor)
write_dataset(fid, "acceptor_channel_bg", ccd_offset_acceptor)
close(fid)

mean_d_1 = mean(donor_channel[findall(x-> x == 1, binned_trajectory)])
std_d_1 = std(donor_channel[findall(x-> x == 1, binned_trajectory)])

mean_d_2 = mean(donor_channel[findall(x-> x == 2, binned_trajectory)])
std_d_2 = std(donor_channel[findall(x-> x == 2, binned_trajectory)])

mean_a_1 = mean(acceptor_channel[findall(x-> x == 1, binned_trajectory)])
std_a_1 = std(acceptor_channel[findall(x-> x == 1, binned_trajectory)])

mean_a_2 = mean(acceptor_channel[findall(x-> x == 2, binned_trajectory)])
std_a_2 = std(acceptor_channel[findall(x-> x == 2, binned_trajectory)])

donor_channel = zeros(Integer, nbins)
acceptor_channel = zeros(Integer, nbins)
for bin in 1:nbins
	if binned_trajectory[bin] == 1

		donor_channel[bin] = Integer(
			round(rand(Normal(mean_d_1, std_d_1))))
		acceptor_channel[bin] = Integer(
			round(rand(Normal(mean_a_1, std_a_1))))

	elseif binned_trajectory[bin] == 2

		donor_channel[bin] = Integer(
			round(rand(Normal(mean_d_2, std_d_2))))
		acceptor_channel[bin] = Integer(
			round(rand(Normal(mean_a_2, std_a_2))))
	end
end

plot_donor = plot(dt .* collect(0:nbins-1), donor_channel,
	xlabel="Time (s)" , ylabel="Intensity (ADU)",
	legend = false, linewidth= 1.5,
	xlims = (-0.1, dt*nbins),
#	xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
#		ylims = (0.7,n_system_states),
# 	yticks = (collect(1.0:convert(Float64, n_system_states)),
# 	collect(1:n_system_states)),
	xtickfontsize=18, ytickfontsize=18,
	xguidefontsize = 20, yguidefontsize = 20,
	fontfamily = "Computer Modern",
	right_margin=5mm, bottom_margin = 12mm,
	left_margin = 10mm, color = :green,
	dpi = 600, format = :svg,
	size=(1200, 300))

plot_acceptor = plot!(dt .* collect(0:nbins-1), acceptor_channel,
	xlabel="Time (s)" , ylabel="Intensity (ADU)",
	legend = false, linewidth= 1.5,
	xlims = (-0.1, dt*nbins),
#	xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
#		ylims = (0.7,n_system_states),
# 	yticks = (collect(1.0:convert(Float64, n_system_states)),
# 	collect(1:n_system_states)),
	xtickfontsize=18, ytickfontsize=18,
	xguidefontsize = 20, yguidefontsize = 20,
	fontfamily = "Computer Modern",
	right_margin=5mm, bottom_margin = 12mm,
	left_margin = 10mm, color = :red,
	dpi = 600, format = :svg,
	size=(1200, 300))

plot_a = plot(transition_time, system_state,
	xlabel="Time (s)" , ylabel="State",
	legend = false, linewidth= 1.5, linetype=:steppost,
 	xlims = (-0.1, dt*nbins),
#	xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
#		ylims = (0.7,n_system_states),
 	yticks = (collect(1.0:convert(Float64, n_system_states)),
 	collect(1:n_system_states)),
	xtickfontsize=18, ytickfontsize=18,
	xguidefontsize = 20, yguidefontsize = 20,
	fontfamily = "Computer Modern",
	right_margin=5mm, bottom_margin = 12mm,
	left_margin = 10mm, color = :blue,
	dpi = 600, format = :svg,
	size=(1200, 300))

plot_b = plot(dt .* collect(0:nbins-1), binned_trajectory,
	xlabel="Time (s)" , ylabel="State",
	legend = false, linewidth= 1.5, linetype=:steppost,
	xlims = (-0.1, dt*nbins),
#	xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
#		ylims = (0.7,n_system_states),
 	yticks = (collect(1.0:convert(Float64, n_system_states)),
 	collect(1:n_system_states)),
	xtickfontsize=18, ytickfontsize=18,
	xguidefontsize = 20, yguidefontsize = 20,
	fontfamily = "Computer Modern",
	right_margin=5mm, bottom_margin = 12mm,
	left_margin = 10mm, color = :blue,
	dpi = 600, format = :svg,
	size=(1200, 300))

file_name = string(working_directory, "test_trace_gaussian.h5")
fid = h5open(file_name,"w")
write_dataset(fid, "donor_channel", convert.(Int64, donor_channel))
write_dataset(fid, "acceptor_channel", convert.(Int64, acceptor_channel))
write_dataset(fid, "donor_channel_bg", ccd_offset_donor)
write_dataset(fid, "acceptor_channel_bg", ccd_offset_acceptor)
close(fid)


file_name = string(working_directory, "gt_test_trace_gaussian.h5")
fid = h5open(file_name,"w")
write_dataset(fid, "binned_trajectory", convert.(Int64, binned_trajectory))
write_dataset(fid, "mean_donor_1", mean_d_1)
write_dataset(fid, "mean_donor_2", mean_d_2)
write_dataset(fid, "mean_acceptor_1", mean_a_1)
write_dataset(fid, "mean_acceptor_2", mean_a_2)
write_dataset(fid, "std_donor_1", std_d_1)
write_dataset(fid, "std_donor_2", std_d_2)
write_dataset(fid, "std_acceptor_1", std_a_1)
write_dataset(fid, "std_acceptor_2", std_a_2)
close(fid)

histogram(donor_channel, bin = 80)


l = @layout [a; b]
plot(plot_a, plot_b, size=(1200,600), layout=l)


#mean_d_1 = 10
#mean_a_1 = 7
#
#mean_d_2 = 12
#mean_a_2 = 6


donor_channel = zeros(Integer, nbins)
acceptor_channel = zeros(Integer, nbins)
da_crosstalk = 0.0
for bin in 1:nbins
	if binned_trajectory[bin] == 1

		donor_channel[bin] = Integer(
			rand(Poisson(mean_d_1 - da_crosstalk*mean_d_1)))
		acceptor_channel[bin] = Integer(
			rand(Poisson(mean_a_1 + da_crosstalk*mean_d_1)))

	elseif binned_trajectory[bin] == 2

		donor_channel[bin] = Integer(
			rand(Poisson(mean_d_2 - da_crosstalk*mean_d_2)))
		acceptor_channel[bin] = Integer(
			rand(Poisson(mean_a_2 + da_crosstalk*mean_d_2)))
	end
end

plot_donor = plot(dt .* collect(0:nbins-1), donor_channel,
	xlabel="Time (s)" , ylabel="Intensity (ADU)",
	legend = false, linewidth= 1.5,
	xlims = (-0.1, dt*nbins),
#	xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
#		ylims = (0.7,n_system_states),
# 	yticks = (collect(1.0:convert(Float64, n_system_states)),
# 	collect(1:n_system_states)),
	xtickfontsize=18, ytickfontsize=18,
	xguidefontsize = 20, yguidefontsize = 20,
	fontfamily = "Computer Modern",
	right_margin=5mm, bottom_margin = 12mm,
	left_margin = 10mm, color = :green,
	dpi = 600, format = :svg,
	size=(1200, 300))

plot_acceptor = plot!(dt .* collect(0:nbins-1), acceptor_channel,
	xlabel="Time (s)" , ylabel="Intensity (ADU)",
	legend = false, linewidth= 1.5,
	xlims = (-0.1, dt*nbins),
#	xticks = 1:Int(ceil((current_draw-1)/5)):current_draw,
#		ylims = (0.7,n_system_states),
# 	yticks = (collect(1.0:convert(Float64, n_system_states)),
# 	collect(1:n_system_states)),
	xtickfontsize=18, ytickfontsize=18,
	xguidefontsize = 20, yguidefontsize = 20,
	fontfamily = "Computer Modern",
	right_margin=5mm, bottom_margin = 12mm,
	left_margin = 10mm, color = :red,
	dpi = 600, format = :svg,
	size=(1200, 300))

file_name = string(working_directory, "test_trace_poissonian_high_snr.h5")
fid = h5open(file_name,"w")
write_dataset(fid, "donor_channel", convert.(Int64, donor_channel))
write_dataset(fid, "acceptor_channel", convert.(Int64, acceptor_channel))
write_dataset(fid, "donor_channel_bg", ccd_offset_donor)
write_dataset(fid, "acceptor_channel_bg", ccd_offset_acceptor)
close(fid)

file_name = string(working_directory, "gt_test_trace_poissonian_high_snr.h5")
fid = h5open(file_name,"w")
write_dataset(fid, "binned_trajectory", convert.(Int64, binned_trajectory))
write_dataset(fid, "mean_donor_1", mean_d_1)
write_dataset(fid, "mean_donor_2", mean_d_2)
write_dataset(fid, "mean_acceptor_1", mean_a_1)
write_dataset(fid, "mean_acceptor_2", mean_a_2)
close(fid)
