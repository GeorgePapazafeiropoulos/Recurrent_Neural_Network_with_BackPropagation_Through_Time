%% Local function: example_generate_resonance_enhanced_noise
function noise_signal = example_generate_resonance_enhanced_noise(n_samples, dt, m, k, c)
% Generate white noise with enhanced power at the SDOF resonance region
% This ensures RNN sees resonance amplification during training
Fs = 1/dt;
f_n = sqrt(k/m) / (2*pi);  % Natural frequency
zeta = c / (2*sqrt(m*k));  % Damping ratio
% Design frequency weighting filter
% Create filter that boosts frequencies near resonance
% Frequency vector
f = Fs * (0:floor(n_samples/2)) / n_samples;
% SDOF transfer function magnitude (squared for power)
w = 2*pi*f;
H_mag2 = 1 ./ ((k - m*w.^2).^2 + (c*w).^2);  % |H(w)|^2
% Normalize and apply as frequency weighting
H_weight = sqrt(H_mag2 / max(H_mag2));  % Square root for amplitude
% Ensure it's symmetric for real signal
if mod(n_samples, 2) == 0
    H_weight = [H_weight, fliplr(H_weight(2:end-1))];
else
    H_weight = [H_weight, fliplr(H_weight(2:end))];
end
% Generate white noise
white_noise = randn(n_samples, 1);
% Apply FFT
X = fft(white_noise);
% Apply resonance weighting
X_weighted = X .* H_weight(:);
% Inverse FFT
colored_noise = real(ifft(X_weighted));
% Normalize
colored_noise = colored_noise - mean(colored_noise);
colored_noise = colored_noise / std(colored_noise);
% Scale to desired amplitude
noise_signal = 10.0 * colored_noise;
end

