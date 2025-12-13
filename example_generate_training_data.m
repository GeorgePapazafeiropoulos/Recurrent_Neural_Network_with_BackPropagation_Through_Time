%% Local function: example_generate_training_data
function [input_data, output_data] = example_generate_training_data(n_samples,...
    dt, m, k, c)
% Function for generation of training data (white noise)
input_data = example_generate_resonance_enhanced_noise(n_samples, dt, m, k, c);
% Compute SDOF response
output_data = example_compute_sdof_response(input_data, dt, m, k, c);
end

