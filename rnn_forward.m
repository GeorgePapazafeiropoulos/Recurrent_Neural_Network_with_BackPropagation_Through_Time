%% Local function: rnn_forward
function [h, outputs, hidden_states] = rnn_forward(X, W1, W2, W3, b1, b2)
% Forward pass of RNN
[batch_size, seq_len] = size(X);
hidden_size = size(W1, 1);
% Initialize
h = zeros(hidden_size, batch_size, seq_len);
outputs = zeros(batch_size, seq_len);
hidden_states = zeros(hidden_size, batch_size, seq_len + 1);
for t = 1:seq_len
    if t == 1
        prev_h = zeros(hidden_size, batch_size);
    else
        prev_h = h(:, :, t-1);
    end
    % RNN cell with numerical stability
    x_t = X(:, t)';
    input_part = W1 * x_t;
    hidden_part = W2 * prev_h;
    % Tanh activation
    h(:, :, t) = tanh(input_part + hidden_part + b1);
    hidden_states(:, :, t) = h(:, :, t);
    % Output: linear activation for regression
    outputs(:, t) = (W3 * h(:, :, t) + b2)';
end
hidden_states(:, :, end) = h(:, :, end);
end

