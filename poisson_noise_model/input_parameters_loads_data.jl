working_directory = string("/home/singularity/Dropbox (ASU)/bayesianCodes/julia/multiFRET/BNP-FRET_binned/poissonian/")
file_prefix = string("test_trace_gaussian")

println(" Importing FRET Traces...")
flush(stdout);

const donor_channel_data::Vector{Float64},
acceptor_channel_data::Vector{Float64},
donor_channel_bg::Float64,
acceptor_channel_bg::Float64,
nbins::Int64 = get_data()

println(" Done.")
flush(stdout);

println(" *********************************************************")
#println(" file_id, = ", file_id)
println(" nbins = ", nbins)
println(" *********************************************************")
flush(stdout);

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

const detection_eff_acceptor::Float64 = 0.9
const da_crosstalk::Float64 = 0.0

const starting_temperature::Float64 = 100000.0
const max_n_system_states::Int64 = 2
const expected_n_system_states::Int64 = 2
const modeling_choice = "parametric"
