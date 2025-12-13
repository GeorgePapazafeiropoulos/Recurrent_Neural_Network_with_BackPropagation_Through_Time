%% Local function: rnn_backward
function [dW1, dW2, dW3, db1, db2] = rnn_backward(...
    X, y, h, outputs, W1, W2, W3, b1, b2, batch_size, window_length)
% Function for backward pass of RNN
[~, seq_len] = size(X);
hidden_size = size(W1, 1);
dW1 = zeros(size(W1));
dW2 = zeros(size(W2));
dW3 = zeros(size(W3));
db1 = zeros(size(b1));
db2 = zeros(size(b2));
% Only compute gradients for last time step (many-to-one)
t = seq_len;
% Gradient of loss w.r.t. outputs
dy = 2 * (outputs(:, t) - y) / batch_size;
% Output layer gradients
h_t = squeeze(h(:, :, t));
dW3 = dy' * h_t';
db2 = sum(dy, 1)';
% Backpropagate through hidden states
dh = W3' * dy';
% BPTT loop
% Limit BPTT to prevent vanishing gradients
for t = seq_len:-1:max(1, seq_len-window_length)
    h_t = squeeze(h(:, :, t));
    % Gradient through tanh
    dtanh = (1 - h_t.^2) .* dh;
    % Previous hidden state
    if t == 1
        prev_h = zeros(hidden_size, batch_size);
    else
        prev_h = squeeze(h(:, :, t-1));
    end
    % Current input
    x_t = X(:, t)';
    % Accumulate gradients
    dW1 = dW1 + (dtanh * x_t') / batch_size;
    dW2 = dW2 + (dtanh * prev_h') / batch_size;
    db1 = db1 + sum(dtanh, 2) / batch_size;
    % Gradient for previous time step
    if t > 1
        dh = W2' * dtanh;
        % Gradient clipping
        dh_norm = sqrt(sum(dh(:).^2));
        if dh_norm > 1.0
            dh = dh / dh_norm;
        end
    end
end
% Gradient clipping
max_grad = 1.0;
dW1 = max(min(dW1, max_grad), -max_grad);
dW2 = max(min(dW2, max_grad), -max_grad);
dW3 = max(min(dW3, max_grad), -max_grad);
end

