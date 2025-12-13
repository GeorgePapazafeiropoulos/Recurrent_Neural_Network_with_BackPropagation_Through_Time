%% Local function: rnn_initialize_weights
function [W1, W2, W3, b1, b2] = rnn_initialize_weights(input_size, hidden_size, output_size)
% Use larger initialization to prevent small outputs
W1 = 0.1 * randn(hidden_size, input_size);
W2 = 0.1 * randn(hidden_size, hidden_size);
W3 = 0.5 * randn(output_size, hidden_size);
% Biases
b1 = zeros(hidden_size, 1);
b2 = 0.1 * ones(output_size, 1);  % Positive bias to encourage larger outputs
end

