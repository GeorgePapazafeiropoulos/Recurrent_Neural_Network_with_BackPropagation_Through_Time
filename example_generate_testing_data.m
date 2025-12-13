%% Local function: example_generate_testing_data
function [input_data, output_data] = example_generate_testing_data(n_samples,...
    dt, m, k, c)
% Generation of testing Data (sum of sinusoids)
% Generate input excitation with specific frequencies
% These frequencies should be within the training frequency band
t = (0:n_samples-1)' * dt;
% Calculate natural frequency
f_n = sqrt(k/m) / (2*pi);
% Test with sinusoids at various frequencies (including resonance)
% Choose frequencies that the RNN should have learned from white noise
test_frequencies = [0.5 * f_n, f_n, 2 * f_n, 3 * f_n];  % Below, at, and above resonance
% Create multi-sine signal
input_data = zeros(n_samples, 1);
amplitudes = [0.8, 0.5, 0.3, 0.2];  % Different amplitudes
phases = [0, pi/4, pi/2, 3*pi/4];   % Different phases
for i = 1:length(test_frequencies)
    input_data = input_data + amplitudes(i) * sin(2 * pi * test_frequencies(i) * t + phases(i));
end
% Add small noise to make it more realistic
input_data = input_data + 0.05 * randn(n_samples, 1);
% Scale to similar amplitude as training
input_data = 10.0 * input_data / std(input_data);
% Compute SDOF response
output_data = example_compute_sdof_response(input_data, dt, m, k, c);
end

